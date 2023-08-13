"""
Training on multiple GPUs
"""
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch
import atexit

def ddp_setup(rank: int, world_size: int):
    """
    Using a master port and master address sometimes work, but 
    not on all machines like using `MASTER_ADDR` and `MASTER_PORT` 
    """
    init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        init_method="file:///tmp/peers"
    )

def run_on_multiple_gpus(gpu_call, *args):
    world_size = torch.cuda.device_count()
    output = mp.spawn(gpu_call, args=(
        world_size,
    ) + args, nprocs=world_size)
    return output

def cleanup():
    try:
        destroy_process_group()
    except Exception as e:
        print("Failed to cleanup destroy")

# destroys the process group
atexit.register(cleanup)
