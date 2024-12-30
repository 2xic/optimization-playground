import time

class Checkpoint:
    def __init__(self):
        self.last_checkpoint = time.time()

    def checkpoint(self):
        current_time = time.time()
        delta = current_time - self.last_checkpoint
        # checkpoint every 5 minutes
        checkpoint = delta > 60 * 5
        if checkpoint:
            self.last_checkpoint = time.time()
        return checkpoint
