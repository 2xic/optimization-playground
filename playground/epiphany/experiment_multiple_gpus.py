"""
Training on multiple GPUs
"""
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.distributed.MultipleGPUsTrainingLoop import MultipleGPUsTrainingLoop
import torch.optim as optim
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import json

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# gpu_id (called rank in the docs) is automatically set by mp spawn
def main(gpu_id, world_size):
    y_train_accuracy, y_test_accuracy, x = [], [], []
    ddp_setup(gpu_id, world_size)
    HOURS = 8
    model = BasicConvModel(n_channels=3)
    optimizer = optim.Adam(model.parameters())
    train, test = get_dataloader(
        shuffle=False,
        sampler=lambda dataset: DistributedSampler(dataset),
    )
    trainer = MultipleGPUsTrainingLoop(
        model,
        optimizer,
        gpu_id=gpu_id
    )
    start = time.time()
    while (time.time() - start) < (60 * 60) * HOURS:
        (_, acc) = trainer.train(
            train
        )
        if gpu_id == 0 and trainer.epoch % 10 == 0:
            x.append(trainer.epoch)
            y_train_accuracy.append(acc.item())
            accuracy = trainer.eval(test)
            y_test_accuracy.append(accuracy.item())
        
            with open("results.json", "w") as file:
                file.write(json.dumps({
                    "x": x,
                    "training": y_train_accuracy,
                    "testing": y_test_accuracy
                }))
    print("Done")
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(
        world_size,
    ), nprocs=world_size)
