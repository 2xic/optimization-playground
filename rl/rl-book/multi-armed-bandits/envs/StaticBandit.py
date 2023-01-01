import random
import numpy as np

class StaticBandit:
    def __init__(self, k, base_line = 0, scale=None) -> None:
        base_reward = list(map(lambda x: x + random.randint(-5, 5), list(range(k))))
        random.shuffle(base_reward)
        if scale is None:
            scale = k
        base_reward = list(map(lambda x: x/ scale, base_reward))
        self.rewards = base_reward
        self.optimal_action = np.argmax(self.rewards)
        self.base_line = base_line

    def __call__(self, action):
        return self.base_line + self.rewards[action]
