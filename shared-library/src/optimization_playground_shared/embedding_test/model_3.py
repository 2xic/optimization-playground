from .base_model import AccumulateLoss
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
from .checkpoints import Checkpoint
from .model_config import ModelConfig
import torch

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig("TfIdfAnchor")

    def get_training_parameters(self):
        return self.model_config.get_model_parameters()

    def train(self, device):
        self.model_config.train(device)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
