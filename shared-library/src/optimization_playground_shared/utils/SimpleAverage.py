import numpy as np

class SimpleAverage:
    def __init__(self):
        self.arr = None
        self.N = 0

    def add(self, arr):
        if type(arr) == list:
            arr = np.asarray(arr)
        
        if self.arr is None:
            self.arr = arr
        else:
            self.arr += arr
        self.N += 1

    def res(self):
        return (self.arr / self.N).tolist()
