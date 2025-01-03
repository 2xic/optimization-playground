import torch.utils
import torch.utils.data
from .loss_functions import MinimalCrossEntropyLoss, SimpleContrastiveLoss, NegativeSample, NextTokenPrediction
import torch.nn as nn
from .dataloader import get_dataloader
from ..nlp.DocumentEncoderSequence import SimpleVocab
import torch.optim as optim
from optimization_playground_shared.process_pools.MultipleGpus import is_main_gpu, get_rank
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import ModelParameters
from .model_variants import SimpleEmbeddingModelWrapper, HighLevelModel, TransformerModelWrapper
import torch
from .checkpoints import Checkpoint
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from .checkpoints import Checkpoint
from tqdm import tqdm
from .loss_functions import TfIdfAnchor
import random
from optimization_playground_shared.utils.RunningAverage import RunningAverage
from .evals import EvaluationMetrics
import json
import time
import torch.profiler
import cProfile
import pstats
import sys

class ModelConfig:
    def __init__(self, loss):
        # TODO: what I actually want is to have the dataloader and loss coupled
        self.loss_dataloader = {
            "TripletMarginLoss": "triplet_loss",
            "NextTokenPrediction": "next_token_prediction"
        }
        self._model = None
        self.loss_name = loss  
        self.metrics_tracker = None
        self.checkpoint = Checkpoint(
            timeout_minutes=30
        )
        self.model_parameters = self._get_model_parameters()
        self.is_main_gpu = is_main_gpu()

    def train(self, device: torch.device):
        self.device = device
        self._model._device = self.device
        main_gpu = is_main_gpu()
        if main_gpu:
            self.metrics_tracker = Tracker("model_" + self.loss_name) 
            self.evaluationMetrics = EvaluationMetrics()
            print("created the metrics tracker")
        self.model_parameters.model.to(device)
        self.model_parameters.model.train()

#        self.model_parameters.dataloader.batch_size = self._find_batch_size()
#        print(f"Batch size {self.model_parameters.dataloader.batch_size}")

        epoch = 0
        progress = tqdm() if self.metrics_tracker is not None else None
        counter = 0
        training_accuracy = None
        prediction = None
        batch_start_time = time.time() 
        """
        profiler = cProfile.Profile()
        profiler.enable()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True
            ) as prof:
        """
        while not self.checkpoint.timeout():
            avg_loss = RunningAverage()
            if self.metrics_tracker is not None:
                training_accuracy = self.evaluationMetrics.eval(self._model)
            if epoch > 0:
                stats = {
                    "batch_per_second": (counter / (time.time() - batch_start_time)),
                    "training_accuracy": training_accuracy,
                }
                prediction = Prediction.text_prediction(json.dumps(stats))
            for _, args in enumerate(self.model_parameters.dataloader):
                loss = self.forward(*args)
                avg_loss.update(loss.item())
                loss.backward()
                self.update()
                if self.metrics_tracker is not None:
                    self.metrics_tracker.queue(Metrics(
                        epoch,
                        loss=avg_loss.value,
                        training_accuracy=training_accuracy,
                        prediction=prediction,
                    ))
                    progress.set_description(f"Loss: {avg_loss.value:.4f}, Accuracy {training_accuracy}, batch_size {args[0].shape[0]}")
                    progress.update(1)
                # Checkpoint.
                counter = (counter + 1) % 32
                # if self.checkpoint.checkpoint():
                #     self._model.save()
            epoch += 1
        """
            break
        if is_main_gpu():
            prof.export_chrome_trace("profile_results.json")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(25)
        """
    def forward(self, *args) -> torch.Tensor:
        if self.loss_name == "TfIdfAnchor":
            documents = random.sample(self.model_parameters.dataloader.docs, 8)
            model_embeddings = self._model._get_embedding(documents, device=self.device)
            return self.model_parameters.loss(model_embeddings, documents)
        elif self.loss_name != "TripletMarginLoss":
            (x, y) = args
            x = x.to(self.device)
            y = y.to(self.device)
            return self.model_parameters.loss(self.model_parameters.model(x), y)
        else:
            (x, y, z) = args
            x = x.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)
            return self.model_parameters.loss(self.model_parameters.model(x), self.model_parameters.model(y), self.model_parameters.model(z))
        
    def update(self):
        self.model_parameters.optimizer.step()
        self.model_parameters.optimizer.zero_grad()

    def get_model_parameters(self):
        return self.model_parameters
    
    def load_trained_model(self):
        model = self._get_model(SimpleVocab())
        return model.load()

    def _get_model(self, vocab: SimpleVocab) -> HighLevelModel:
        if self.loss_name == "NextTokenPrediction":
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
        document_encoder: SimpleVocab = SimpleVocab().load(
            ".", 
            prefix="pretrained"
        )
        dataloader = get_dataloader(document_encoder, self.loss_dataloader.get(self.loss_name, None))

        #gpu_id = get_rank()
        #size = len(dataloader.docs) // torch.cuda.device_count() if torch.cuda.device_count() > 0 else len(dataloader.docs)
        #dataloader.docs = dataloader.docs[size * gpu_id:(gpu_id + 1) * size] 
        #assert len(dataloader.docs) > 0

        model = self._get_model(document_encoder).create_model()
        optimizer =  optim.Adam(model.model.parameters())
        self._model = model
        loss_lookup = {
            "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
            "SimpleContrastiveLoss": SimpleContrastiveLoss(),
            "NegativeSample": NegativeSample(),
            "NextTokenPrediction": NextTokenPrediction(),
            "TripletMarginLoss": nn.TripletMarginLoss(),
#            "TfIdfAnchor": TfIdfAnchor(dataloader.docs),
        }
        if self.loss_name == "NextTokenPrediction":
            dataloader.SEQUENCE_LENGTH = 128

        return ModelParameters(
            loss=loss_lookup[self.loss_name],
            optimizer=optimizer,
            model=model.model,
            dataloader=dataloader,
        )
