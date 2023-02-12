from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

class ProcessPool:
    def __init__(self, max_workers=None):
        mp.set_start_method('spawn')
        # From the docs it should default to a reasonable number https://docs.python.org/3/library/concurrent.futures.html
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    def execute(self, function, inputs):
        lock = mp.Manager().Lock()
        for i in self.executor.map(function, [lock,]*len(inputs), inputs):
            yield i
        self.executor.shutdown()
