import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, actions, eps, decay=1, search=None) -> None:
        self.actions = actions
        self.eps = eps
        self.decay = decay
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
            self.eps *= self.decay # 0.01
            return self.search()
