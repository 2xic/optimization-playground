"""
Training on multiple GPUs
"""
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch
import atexit
import os
import hashlib
import sys

def get_target_port() -> int:
    main_module = sys.modules['__main__']
    data = main_module.__file__

    min_port: int = 10000 
    max_port: int = 65535

    # Hash the data using SHA-256
    hasher = hashlib.sha256()
    hasher.update(data.encode('utf-8'))
    hash_value = int(hasher.hexdigest(), 16)
    port_range = max_port - min_port + 1
    return (hash_value % port_range) + min_port


def ddp_setup(rank: int, world_size: int):
    """
    Using a master port and master address sometimes work, but 
    not on all machines like using `MASTER_ADDR` and `MASTER_PORT` 
    """
    if os.path.isfile("/tmp/peers"):
        print(f"Note that peer exists :) ({rank})")
    os.environ["MASTER_ADDR"] = "localhost"
    # Generate ports so we can have multiple instances running at the same time.
    os.environ["MASTER_PORT"] =  str(int(get_target_port()))

    init_process_group(
        backend="nccl",
        rank=rank, 
        world_size=world_size,
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
