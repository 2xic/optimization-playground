import time
import torch.distributed as dist

class Checkpoint:
    def __init__(self, timeout_minutes):
        self.start_time = time.time()
        self.last_checkpoint = time.time()
        self.timeout_minutes = timeout_minutes

    def checkpoint(self):
        current_time = time.time()
        delta = current_time - self.last_checkpoint
        # checkpoint every 5 minutes
        checkpoint = delta > 60 * 5
        if checkpoint:
            self.last_checkpoint = time.time()
        
        return checkpoint and self.is_main_gpu()

    def timeout(self):
        current_time = time.time()
        delta = current_time - self.start_time
        # checkpoint every n minutes
        checkpoint = delta > 60 * self.timeout_minutes
        if checkpoint:
            self.last_checkpoint = time.time()
        return checkpoint

    def is_main_gpu(self):
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0
