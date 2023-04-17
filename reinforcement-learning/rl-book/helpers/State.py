from .StateValue import StateValue
import numpy as np

class State:
    def __init__(self, n, value_constructor=lambda n: StateValue(n, initial_value=np.random.rand)) -> None:
        self.state = {}
        self.value_constructor = value_constructor
        self.n = n

    def __getitem__(self, key):
        if not key in self.state:
            self.state[key] = self.value_constructor(self.n)
        return self.state[key]

    def __setitem__(self, key, value):
        self.state[key] = value
