import numpy as np

class SimpleRlEnv:
    def __init__(self) -> None:
        self.reset()
        self.level_length = 1_0
        
    def reset(self):
        self.state = np.zeros((2))
        self.index = 0
        self.state[self.index] = 1

    def step(self, action):
        relative_index = self.index % 2
        observation, reward, terminated, truncated, info = (
            self.state,
            int(action == relative_index),
            (self.index >= self.level_length), 
            False,
            False
        )
        self.index += 1
        self.state[relative_index] = 1
        self.state[(relative_index + 1) % 2] = 0

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )    

