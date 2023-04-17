import random

class RandomAgent:
    def __init__(self, **kwargs) -> None:
        pass

    def train(self, env):
        while not env.done:
            env.play(self.action(env))

    def action(self, env):
        return  env.play(random.sample(env.legal_actions, k=1)[0])
