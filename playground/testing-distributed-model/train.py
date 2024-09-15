"""
Train to predict next pixel from the previous batch of pixels
"""
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from torch.utils.data.distributed import DistributedSampler


class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()

    def get_training_parameters(self):
        model = BasicConvModel(input_shape=(1, 28, 28))
        optimizer = optim.Adam(model.parameters())
        loss = torch.nn.NLLLoss()
        return model, optimizer, loss

    def get_dataloader(self, device):
        train, _ = get_dataloader(
            sampler=lambda dataset: DistributedSampler(dataset, shuffle=True),
            shuffle=False
        )
        return train


if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
