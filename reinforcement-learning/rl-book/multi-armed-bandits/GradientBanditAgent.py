import numpy as np
import torch
from helper.RunningAverage import RunningAverage
import numpy as np
"""
As described in section 2.8
"""

class GradientBanditAgent:
    def __init__(self, K, use_baseline, lr) -> None:
        self.Q = [0, ] * K
        self.N = [0, ] * K
        self.K = K
        self.use_baseline = use_baseline
        self.running_avg = RunningAverage()
        self.lr = lr

    def action(self):
#        action = self.policy(self)
        p = self.p()
        action = np.random.choice(
            self.K,
            p=p
        )
        return action

    def on_policy(self):
        return np.argmax(self.Q)

    def p(self):
        np_arr = np.asarray(self.Q)
        exp = np.exp(np_arr)
        sum_exp = np.sum(exp)

        return (exp / sum_exp)

    def update(self, action, reward):
        if self.running_avg.value is not None:
            for i in range(self.K):
                if i == action:
                    self.Q[i] = self.Q[i] + self.lr * (reward - self.running_avg.value) * (1 - self.p()[i])
                else:
                    self.Q[i] = self.Q[i] - self.lr * (reward - self.running_avg.value) * self.p()[i]

        if not self.use_baseline:
            self.running_avg.update(0)
        else:
            self.running_avg.update(reward) 

