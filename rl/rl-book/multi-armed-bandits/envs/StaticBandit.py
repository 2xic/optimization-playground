import random
import numpy as np

class StaticBandit:
    def __init__(self, k) -> None:
        base_reward = list(range(k))
        random.shuffle(base_reward)
        base_reward = list(map(lambda x: x/ k, base_reward))
        self.rewards = base_reward
        self.optimal_action = np.argmax(self.rewards)

    def __call__(self, action):
        return self.rewards[action]
