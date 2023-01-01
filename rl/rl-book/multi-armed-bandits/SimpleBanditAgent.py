import numpy as np
"""
As described in section 2.4
"""


class SimpleBanditAgent:
    def __init__(self, K, policy, q_0=0) -> None:
        self.Q = [q_0, ] * K
        self.N = [0, ] * K
        self.K = K
        self.policy = policy

    def action(self):
        action = self.policy(self)
        return action

    def on_policy(self):
        return np.argmax(self.Q)

    def update(self, action, reward, lr=None):
        self.N[action] += 1
        if lr is None:
            lr = (
                1 / self.N[action]
            )
        self.Q[action] += lr * (reward - self.Q[action])
