import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, actions, eps, decay=1, search=None) -> None:
        self.actions = actions
        self.eps = eps
        self.decay = decay
        self.eps_decay_limit = 0.01
        self.search = (lambda : random.randint(0, self.actions - 1))\
                                if search is None\
                                else search
        

    def __call__(self, agent):
        if np.random.rand() >= self.eps :
            return agent.on_policy()
        else:
            if self.eps == 0:
                raise Exception("Something is wrong")
            # decay
            if self.eps > self.eps_decay_limit:
                self.eps *= self.decay
            return self.search()
