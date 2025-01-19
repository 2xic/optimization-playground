import torch.utils
import torch.utils.data
from .loss_functions import MinimalCrossEntropyLoss, SimpleContrastiveLoss, NegativeSample, NextTokenPrediction
import torch.nn as nn
from .dataloader import get_dataloader
from ..nlp.DocumentEncoderSequence import SimpleVocab
import torch.optim as optim
from optimization_playground_shared.process_pools.MultipleGpus import is_main_gpu
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import ModelParameters
from .model_variants import SimpleEmbeddingModelWrapper, HighLevelModel, TransformerModelWrapper
import torch
from .checkpoints import Checkpoint
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from .checkpoints import Checkpoint
from tqdm import tqdm
import random
from optimization_playground_shared.utils.RunningAverage import RunningAverage
from .evals import EvaluationMetrics
import json
import time
import torch.profiler
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TimerContext:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        delta = time.time() - self.time
        logger.debug(f"time {self.name}: {delta}")

        if exc_type:
            raise exc_value
        return False
   
def free_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a
    return f

class ModelConfig:
    def __init__(self, loss):
        # TODO: what I actually want is to have the dataloader and loss coupled
        self.loss_dataloader = {
            "TripletMarginLoss": "triplet_loss",
            "NextTokenPrediction": "next_token_prediction",
            "NextTokenPrediction_post_training": "triplet_loss",
        }
        self._model: Optional[HighLevelModel] = None
        self.loss_name = loss
        self.metrics_tracker = None
        self.evaluation_metrics = None
        self.checkpoint = Checkpoint(
            timeout_minutes=120 * 12
        )
        with TimerContext("loading model parameters"):
            self.model_parameters = self._get_model_parameters()
        self.is_main_gpu = is_main_gpu()
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, device: torch.device):
        print("Ready to train the model.")
        self.device = device
        self._model._device = self.device
        if self.is_main_gpu:
            self.metrics_tracker = Tracker("model_" + self.loss_name) 
            self.evaluation_metrics = EvaluationMetrics()
            print("created the metrics tracker")

        with TimerContext("moving model to gpu"):
            self.model_parameters.model.to(device)
            self.model_parameters.model.train()

        epoch = 0
        progress = tqdm() if self.is_main_gpu else None
        counter = 0
        training_accuracy = None
        prediction = None
        batch_start_time = time.time() 

        logging.debug("Starting to train model")
        logging.debug(f"Free memory before training {free_memory()}")
        while not self.checkpoint.timeout():
            avg_loss = RunningAverage()
            if self.metrics_tracker is not None:
                logger.debug("doing evaluation ... ")
                with TimerContext("evaluation metrics"):
                    training_accuracy = self.evaluation_metrics.eval(self._model)
            if epoch > 0:
                stats = {
                    "batch_per_second": (counter / (time.time() - batch_start_time)),
                    "training_accuracy": training_accuracy,
                }
                if self.loss_name == "NextTokenPrediction":
                    X, y = next(iter(self.model_parameters.dataloader))
                    stats["X"] = self._model.document_encoder.decode(X[0].tolist())
                    stats["y"] = self._model.document_encoder.decode(y[0].tolist())
                prediction = Prediction.text_prediction(json.dumps(stats))
                counter = 0
            for _, args in enumerate(self.model_parameters.dataloader):
                # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                with torch.cuda.amp.autocast():
                    loss = self.forward(*args)
                if not torch.isnan(loss):
                    avg_loss.update(loss.item())
                    #loss.backward()
                    self.scaler.scale(loss).backward()
                    self.update()
                
                if self.metrics_tracker is not None:
                    self.metrics_tracker.queue(Metrics(
                        epoch,
                        loss=avg_loss.value,
                        training_accuracy=training_accuracy,
                        prediction=prediction,
                    ))
                if progress is not None:
                    progress.set_description(f"Loss: {avg_loss.value:.4f}, Accuracy {training_accuracy}, batch_size {args[0].shape[0]}")
                    progress.update(1)
                # Checkpoint.
                if self.checkpoint.checkpoint():
                    self._model.save()
                    if self.evaluation_metrics is not None:
                        training_accuracy = self.evaluation_metrics.eval(self._model)
                counter += 1
            epoch += 1
        self._model.save()

    def forward(self, *args) -> torch.Tensor:
        if self.loss_name == "TfIdfAnchor":
            documents = random.sample(self.model_parameters.dataloader.docs, 8)
            model_embeddings = self._model._get_embedding(documents, device=self.device)
            return self.model_parameters.loss(model_embeddings, documents)             
        elif self.loss_name == "NextTokenPrediction_post_training":
            (x, x_pos, x_neg) = args
            x     = x.to(self.device)
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)
            return self.model_parameters.loss(
                self.model_parameters.model(x)    ,
                self.model_parameters.model(x_pos),
                self.model_parameters.model(x_neg),
            )
        elif self.loss_name != "TripletMarginLoss":
            (x, y) = args
            x = x.to(self.device)
            y = y.to(self.device)
            return self.model_parameters.loss(self.model_parameters.model(x), y)
        else:
            (x, x_pos, x_neg) = args
            x = x.to(self.device)
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)

            x     = self.model_parameters.model(x) 
            x_pos = self.model_parameters.model(x_pos)
            x_neg = self.model_parameters.model(x_neg)

            x     = F.normalize(x, p=2, dim=1)
            x_pos = F.normalize(x_pos, p=2, dim=1)
            x_neg = F.normalize(x_neg, p=2, dim=1)

            return self.model_parameters.loss(
                x    ,
                x_pos,
                x_neg,
            )
        
    def update(self):
        torch.nn.utils.clip_grad_value_(self.model_parameters.model.parameters(), clip_value=1.0)
        #self.model_parameters.optimizer.step()
        #self.model_parameters.optimizer.zero_grad()
        self.scaler.step(self.model_parameters.optimizer)
        self.scaler.update()
        self.model_parameters.optimizer.zero_grad()

    def get_model_parameters(self):
        return self.model_parameters
    
    def load_trained_model(self):
        high_level_model = self._model.load()
        high_level_model.model.document_encoder = high_level_model.document_encoder
        self.model_parameters = self._create_model_parameters_from_model(
            high_level_model,
            high_level_model.document_encoder,
        )
        return self

    def _get_model(self, vocab: SimpleVocab) -> HighLevelModel:
        if self.loss_name in ["NextTokenPrediction", "NextTokenPrediction_post_training"]:
            return TransformerModelWrapper(
                self.loss_name,
                vocab,
            )
        else:
            return SimpleEmbeddingModelWrapper(
                self.loss_name,
                vocab,
            )
    
    def _get_model_parameters(self) -> ModelParameters:
        with TimerContext("loading vocab"):
            document_encoder: SimpleVocab = SimpleVocab().load(
                "/root/", 
                prefix="pretrained"
            )
        with TimerContext("loading model"):
            model = self._get_model(document_encoder).create_model()
        return self._create_model_parameters_from_model(
            model,
            document_encoder,
        )

    def _create_model_parameters_from_model(self, model, document_encoder):
        is_transformer = self.loss_name in [
            "NextTokenPrediction",
            "NextTokenPrediction_post_training"
        ]
        batch_size = 1 if self.loss_name == "NextTokenPrediction" else 256
        with TimerContext("loading dataloader"):
            dataloader, dataset = get_dataloader(
                document_encoder, 
                self.loss_dataloader.get(self.loss_name, None),
                batch_size,
            )

        underlying_model = model.model
        self._model = model
        optimizer =  optim.Adam(underlying_model.parameters())
        with TimerContext("loading loss"):
            loss_lookup = {
                "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
                "SimpleContrastiveLoss": SimpleContrastiveLoss(),
                "NegativeSample": NegativeSample(),
                "NextTokenPrediction": NextTokenPrediction(document_encoder.PADDING_IDX),
                "TripletMarginLoss": nn.TripletMarginLoss(),
                "NextTokenPrediction_post_training": nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)),
            }
        if is_transformer:
            dataset.SEQUENCE_LENGTH = 64

        return ModelParameters(
            loss=loss_lookup[self.loss_name],
            optimizer=optimizer,
            model=underlying_model,
            dataloader=dataloader,
        )
