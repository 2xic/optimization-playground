import numpy as np
import random
import torch

class EpsilonGreedy:
    def __init__(self, actions, eps, decay=1) -> None:
        self.eps = eps
        self.decay = decay
        self.eps_decay_limit = 0.01
        self.actions = actions

    def __call__(self):
        if np.random.rand() >= self.eps:
            return None
        else:
            if self.eps == 0:
                raise Exception("Something is wrong")
            # decay
            if self.eps > self.eps_decay_limit:
                self.eps *= self.decay
            return torch.tensor(random.randint(0, self.actions - 1))
