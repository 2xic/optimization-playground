import numpy as np 
"""
As described in section 2.4
"""

class SimpleBanditAgent:
    def __init__(self, K, policy) -> None:
        self.Q = [0, ]* K
        self.N = [0, ]* K
        self.K = K
        self.policy = policy

    def action(self):
        action = self.policy(self)
        return action

    def on_policy(self):
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (
            1 / self.N[action]
        ) * reward - self.Q[action]
