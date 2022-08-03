import torch

"""
Simple environment
"""

class SimpleEnv:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.env:torch.Tensor = torch.tensor([0, 0])
        self.index = 0
        self.env[self.index] = 1
        self.timeout = 10

    def step(self, action):
        env = self.env.clone()
        reward = torch.tensor([0])
        if action == self.index:
            reward = torch.tensor([1])

        self.env[self.index] = 0
        self.index = (self.index + 1) % 2
        self.env[self.index] = 1
        self.timeout -= 1

        return (
            env,
            reward,
            action,
            # gamma = 1 for now
            torch.tensor([1])
        )

    def done(self):
        return self.timeout < 0
        
