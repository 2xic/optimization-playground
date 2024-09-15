"""
Train to predict next pixel from the previous batch of pixels
"""
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.distributed.PipelineDistrubted import MultipleGpuBigModelWrapper
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from torch.utils.data.distributed import DistributedSampler


""""
Running the following should work.

torchrun --nproc-per-node 2 train_across_gpus.py
"""

class Trainer(MultipleGpuBigModelWrapper):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    train, _ = get_dataloader(
        batch_size=1
    )
    trainer = Trainer()
    trainer.start()
    model = BasicConvModel(input_shape=(1, 28, 28))
    trainer.run(
        model,
        next(iter(train))
    )
