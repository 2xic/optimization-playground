import numpy as np
import random

class Epsilon:
    def __init__(self, decay=0.95):
        self.epsilon = 1
        self.decay = decay
        self.epsilon_min = 0.1

    def action(self, actions):
        if np.random.rand() < self.epsilon:
            action = random.sample(actions, 1)[0]

            if self.epsilon_min < self.epsilon:
                self.epsilon *= self.decay
            return action
        return None
