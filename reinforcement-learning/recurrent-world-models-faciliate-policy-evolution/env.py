import gymnasium as gym
from PIL import Image 
import torch

class Env:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5")

    def reset(self) -> torch.Tensor:
        observation, _info = self.env.reset()
        return self._get_torch_tensor(observation)

    def random_play(self):
        observation, _ = self.env.reset()
        action = None
        previous_observation = None
        while True:
            if action is not None and previous_observation is not None:
                yield (
                    self._get_torch_tensor(observation),
                    self._get_torch_tensor(previous_observation),
                    action
                )
            action = self.env.action_space.sample()
            previous_observation = observation
            observation, reward, terminated, _, _ = self.env.step(action)
            if terminated:
                break

    def _get_torch_tensor(self, observation):
        return torch.from_numpy(observation).float().permute(2, 0, 1) / 255

    def save_observation(self, observation):
        data = Image.fromarray(observation)
        data.save('observation.png')
