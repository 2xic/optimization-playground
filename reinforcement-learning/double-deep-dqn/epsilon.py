import numpy as np
import random

class Epsilon:
    def __init__(self):
        self.epsilon = 1
        self.decay = 0.95
        self.epsilon_min = 0.01

    def action(self):
        if np.random.rand() < self.epsilon:
            action = random.randint(0, 1)

            if self.epsilon_min < self.epsilon:
                self.epsilon *= self.decay
            return action

        return None

    def action_from_array(self, actions):
        if np.random.rand() < self.epsilon:
            action = actions[random.randint(0, len(actions) - 1)]
            return action
        return None

    def update(self):
        if self.epsilon_min < self.epsilon:
            self.epsilon *= self.decay
