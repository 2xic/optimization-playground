# ulimit -n 165536
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
from .model_config import ModelConfig

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig("TripletMarginLoss")

    def get_training_parameters(self):
        return self.model_config.get_model_parameters()

    def train(self, device):
        self.model_config.train(device)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
