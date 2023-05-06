"""
Training on multiple GPUs
"""
from typing import Callable
from ..distributed.TrainingLoopDistributed import TrainingLoopDistributed
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

def run_on_multiple_gpus(gpu_call, *args):
    world_size = torch.cuda.device_count()
    output = mp.spawn(gpu_call, args=(
        world_size,
    ) + args, nprocs=world_size)
    return output
