import torch.nn as nn
import torch.optim as optim
from .base_model import BaseModel, AccumulateLoss
from .loss_functions import MinimalCrossEntropyLoss, SimpleContrastiveLoss, NegativeSample
from .dataloader import get_dataloader, TextDataloader
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
import torch
from tqdm import tqdm

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(*[
             nn.Linear(embed_dim, 1024),
             nn.ReLU(),
             nn.Linear(1024, 2048),
             nn.ReLU(),
             nn.Linear(2048, 1024),
             nn.ReLU(),
             nn.Linear(1024, 512),
             nn.Tanh(),
        ])

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        return self.fc(embedded)

class EmbeddingModelOne(BaseModel):
    def __init__(self, loss=None):
        self.loss_functions = {
            "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
            "SimpleContrastiveLoss": SimpleContrastiveLoss(),
            "NegativeSample": NegativeSample(),
        }
        self.loss = None
        if loss is not None:
            self.loss = self.loss_functions[loss]
        super().__init__(self.__class__.__name__)
        self.accumulator = AccumulateLoss()

    def create(self):
        embed_dim = 64
        self.model = SimpleEmbeddingModel(self.document_encoder.size, embed_dim)
        return self

    def train(self, data_loader: TextDataloader):
        lr = 0.001
        num_epochs = 128

        self.model = self.create()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        epoch = 0
        progress = tqdm()

        while not self.checkpoint.timeout():
            for (X, y) in data_loader:
                loss = self.loss(self.model(X), y)
                self.accumulator.update(loss)
                if self.accumulator.done():
                    self.optimizer.zero_grad()
                    self.accumulator.loss.backward()
                    self.optimizer.step()
                    progress.set_description(f"Loss: {self.accumulator.loss.item():.4f}")
                    progress.update(1)
                    self.accumulator.reset()

                if self.checkpoint.timeout():
                    self.save()
                    break
            epoch += 1
        print("Training completed!")
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

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()

    def get_training_parameters(self):
        optimizer = optim.Adam(self.model_wrapper.model.parameters())
        return self.model_wrapper.model, optimizer, self.model_wrapper.loss

    def train(self):
        super().train()
        self.launch()

    def launch(self):
        if self.gpu_id == 0:
            model = self.model_wrapper
            model.save()

    def get_dataloader(self, gpu_id):
        self.model_wrapper = EmbeddingModelOne(
            "NegativeSample",
        )
        dataloader = get_dataloader(self.model_wrapper.document_encoder, None)
        size = len(dataloader.docs) // torch.cuda.device_count()
        dataloader.docs = dataloader.docs[size * gpu_id:(gpu_id + 1) * size] 
        self.model_wrapper.create()
        assert len(dataloader.docs) > 0
        return dataloader
    
    def batch_done(self):
        if self.model_wrapper.checkpoint.checkpoint():
            self.model_wrapper.save()
        if self.model_wrapper.checkpoint.timeout():
            exit(0)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
