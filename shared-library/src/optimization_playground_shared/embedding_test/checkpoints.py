import time
from optimization_playground_shared.process_pools.MultipleGpus import is_main_gpu

class Checkpoint:
    def __init__(self, timeout_minutes, checkpoint_numbers=5):
        self.start_time = time.time()
        self.last_checkpoint = 0
        self.timeout_minutes = timeout_minutes
        self.checkpoint_numbers = checkpoint_numbers

    def checkpoint(self):
        current_time = time.time()
        delta = current_time - self.last_checkpoint
        # checkpoint every n minutes
        checkpoint = delta > 60 * self.checkpoint_numbers
        if checkpoint:
            self.last_checkpoint = time.time()
        
        return checkpoint and is_main_gpu()

    def timeout(self):
        current_time = time.time()
        delta = current_time - self.start_time
        # checkpoint every n minutes
        checkpoint = delta > 60 * self.timeout_minutes
        return checkpoint
