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

class EmbeddingModelThree(BaseModel):
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

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_wrapper = None
        self.checkpoint = Checkpoint()

    def get_training_parameters(self):
        optimizer = optim.Adam(self.model_wrapper.model.parameters())
        return self.model_wrapper.model, optimizer, None
    
    def train(self):
        anchor_loss = TfIdfAnchor(self.dataloader.docs)
        progress = tqdm(range(len(self.dataloader.docs) // 8))
        self.model.to(self.device)
        while not self.checkpoint.checkpoint():
            for _ in progress:
                documents = random.sample(self.dataloader.docs, 8)
                model_embeddings = self.model_wrapper._get_embedding(documents, device=self.device)
                loss = anchor_loss(model_embeddings, documents)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress.set_description(f"Loss: {loss.item():.4f}")

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
        self.model_wrapper = EmbeddingModelThree()
        dataloader = get_dataloader(self.model_wrapper.document_encoder, None)
        size = len(dataloader.docs) // torch.cuda.device_count()
        dataloader.docs = dataloader.docs[size * gpu_id:(gpu_id + 1) * size] 
        self.model_wrapper.create()
        assert len(dataloader.docs) > 0
        return dataloader

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
