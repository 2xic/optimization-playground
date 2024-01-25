from config import Config
import random

class RandomAgent:
    def __init__(self, config: Config, env):
        self.action_space = config.num_actions
        self.env = env

    def play(self):
        self.env.reset()
        terminated = False
        sum_reward = 0
        while not terminated:
            action = random.randint(0, self.action_space)
            (
                _,
                reward,
                terminated,
                _,
                _
            ) = self.env.step(action)
            sum_reward += reward
        return sum_reward
