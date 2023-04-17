# Example 6.7
import numpy as np

class BiasExample:
    def __init__(self) -> None:
        self.reset()
        self.action_space = 2
        self.legal_actions = list(range(self.action_space))

    @property
    def done(self):
        return self.is_done

    def reset(self):
        self.is_done = False
        self.state = [0, 0, 1, 0]
        self.is_left = False
        self.index = 0

    def play(self, action):
        if self.index == 0:
            self.index += 1
            if action == 0:
                self.is_left = False
                self.is_done = True
                self.state = [0, 0, 0, 1]
                return 0
            elif action == 1:
                self.is_left = True
                self.state = [0, 1, 0, 0]
                return 0 
        else:
            self.state = [1, 0, 0, 0]
            self.is_done = True
            self.is_left = True
            return np.random.normal(-0.1, 1)

    def __str__(self) -> str:
        raise Exception("Should not be called")
