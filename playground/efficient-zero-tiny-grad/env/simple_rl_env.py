import numpy as np

class SimpleRlEnv:
    def __init__(self) -> None:
        self.reset()
        self.level_length = 1_0
        self.last_reward = None
        
    def reset(self):
        self.state = np.zeros((2))
        self.index = 0
        self.state[self.index] = 1

    def step(self, action):
        relative_index = self.index % 2
        assert action in [0, 1], action
        assert relative_index in [0, 1], relative_index
        assert sum(self.state.tolist()) == 1
        observation, reward, terminated, truncated, info = (
            self.state,
            int(action == relative_index),
            (self.index >= self.level_length), 
            None,
            None
        )
        self.index += 1
        old_state = self.state.copy()
        self.state[(relative_index + 1) % 2] = 1
        self.state[(relative_index)] = 0
        # Verify that the state is actually updated correctly
        assert self.state[0] != old_state[0], (self.state, old_state, self.index)
        assert self.state[1] != old_state[1], (self.state, old_state, self.index)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )    

