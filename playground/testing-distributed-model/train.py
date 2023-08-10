"""
Train to predict next pixel from the previous batch of pixels
"""
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
# from optimization_playground_shared.dataloaders.Mnist import get_dataloader
# from optimization_playground_shared.nlp.Transformer import TransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.distributed.TrainWrapper import MultipleGpuTrainWrapper
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from torch.utils.data.distributed import DistributedSampler


class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()

    def _get_model_and_optimizer(self):
        model = BasicConvModel(input_shape=(1, 28, 28))
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def _get_dataloader(self, device):
        train, _ = get_dataloader(
            sampler=lambda dataset: DistributedSampler(dataset, shuffle=True),
            shuffle=False
        )
        return train

    def _loss(self):
        return torch.nn.NLLLoss()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
