import numpy as np

class StateValue:
    def __init__(self, n, initial_value) -> None:
        self.value = [initial_value(), ] * n

    def __setitem__(self, key, value):
        self.value[key] = value

    def __getitem__(self, key):
        return self.value[key]
    
    def np(self):
        return np.asarray(self.value)

    def argmax(self):
        return np.argmax(self.np())

    def max(self):
        return self.value[self.argmax()]
    
    def __str__(self):
        return str(self.value)
        