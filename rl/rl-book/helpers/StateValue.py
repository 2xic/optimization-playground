import numpy as np

class StateValue:
    def __init__(self, n) -> None:
        self.value = [np.random.rand(), ] * n

    def __setitem__(self, key, value):
        self.value[key] = value

    def __getitem__(self, key):
        return self.value[key]
    
    def np(self):
        return np.asarray(self.value)

    def argmax(self):
        return np.argmax(self.np())
