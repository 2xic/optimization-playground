import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, actions, eps) -> None:
        self.actions = actions
        self.eps = eps

    def __call__(self, agent):
        if np.random.rand() >= self.eps :
            return agent.on_policy()
        else:
            if self.eps == 0:
                raise Exception("Something is wrong")
            return random.randint(0, self.actions - 1)
