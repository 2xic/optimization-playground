import numpy as np

class SimpleRlEnv:
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self.state = np.zeros((2))
        self.index = 0
        self.state[self.index] = 1

    def step(self, action):
        self.index += 1
        observation, reward, terminated, truncated, info = (
            self.state,
            int(action == self.index),
            False, 
            False,
            False
        )
        self.state[self.index % 2] = 1
        self.state[(self.index + 1) % 2] = 1
        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )    

