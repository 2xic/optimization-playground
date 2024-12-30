import torch.optim as optim
from .base_model import BaseModel
from .loss_functions import TfIdfAnchor
from .dataloader import get_dataloader, TextDataloader
import random
from .hosted_model import create_flask_app
from .model_1 import SimpleEmbeddingModel
from tqdm import tqdm
from .dataloader import get_dataloader, TextDataloader
from .hosted_model import create_flask_app
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
import torch
from .checkpoints import Checkpoint
import torch.nn as nn

class EmbeddingModelFour(BaseModel):
    def __init__(self):
        super().__init__(
            self.__class__.__name__
        )

    def create(self):
        embed_dim = 64
        self.model = SimpleEmbeddingModel(self.document_encoder.size, embed_dim)
        return self

    def _load_model(self, model_data):
        return SimpleEmbeddingModel(
            model_data["vocab_size"],
            model_data["embed_dim"]
        )
    
    def _get_state_dict(self):
         return {
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "embed_dim": self.model.embed_dim,
        }
    
class AccumulateLoss:
    def __init__(self):
        self.counter = 0
        self.loss = None

    def update(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
        self.counter += 1

    def reset(self):
        self.counter = 0
        self.loss = None

    def done(self):
        return self.counter >= 32

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_wrapper = None
        self.checkpoint = Checkpoint()
        self.accumulator = AccumulateLoss()

    def get_training_parameters(self):
        optimizer = optim.Adam(self.model_wrapper.model.parameters())
        return self.model_wrapper.model, optimizer, None
    
    def train(self):
        anchor_loss = nn.TripletMarginLoss()
        progress = tqdm()
        self.model.to(self.device)
        self.model.train()
        while not self.checkpoint.checkpoint():
            for (x, y, z) in self.dataloader:
                x, y, z = x.to(self.device), y.to(self.device),z.to(self.device)
                loss = anchor_loss(
                    self.model(x), 
                    self.model(y), 
                    self.model(z)
                )
                self.accumulator.update(loss)
                if self.accumulator.done():
                    self.optimizer.zero_grad()
                    self.accumulator.loss.backward()
                    self.optimizer.step()
                    progress.set_description(f"Loss: {self.accumulator.loss.item():.4f}")
                    self.accumulator.reset()
                if self.checkpoint.checkpoint():
                    self.model_wrapper.save()
                    break
            break

        self.model_wrapper.save()
        print("Training completed!")
        self.launch()

    def launch(self):
        if self.gpu_id == 0:
            model = self.model_wrapper
            model.save()
            model.load()
            print(self.model_wrapper.transforms(["hello, this is some text"]))
            create_flask_app(model).run(
                port=8081,
                host="0.0.0.0"
            )

    def get_dataloader(self, gpu_id):
        self.model_wrapper = EmbeddingModelFour()
        dataloader = get_dataloader(self.model_wrapper.document_encoder, "triplet_loss")
        size = len(dataloader.docs) // torch.cuda.device_count()
        dataloader.docs = dataloader.docs[size * gpu_id:(gpu_id + 1) * size] 
        self.model_wrapper.create()
        assert len(dataloader.docs) > 0
        return dataloader

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
