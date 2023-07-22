import gymnasium as gym
from PIL import Image 
import torch

class Env:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5", obs_type="grayscale")
        self.action_size = self.env.action_space.n

    def reset(self) -> torch.Tensor:
        observation, _info = self.env.reset()
        return self._get_torch_tensor(observation)
    
    def agent_play(self, agent):
        agent.reset()
        observation, _ = self.env.reset()
        while True:
            old_observation = self._get_torch_tensor(observation)
            action, info = agent.action(
                old_observation
            )
            
            observation, reward, terminated, _, _ = self.env.step(action)

            yield (old_observation, action, reward, info)

            if terminated:
                break

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
        # making it easier for the model to distinguish 
        observation[:, :][observation[:, :] == 87] = 0
        # Player 2
        observation[:, :][observation[:, :] == 147] = 255
        # Player 1
        observation[:, :][observation[:, :] == 148] = 255

        tensor = torch.from_numpy(observation).float() / 255 
        #.permute(2, 0, 1) / 255
        # return tensor[:, 34:-16]
        return tensor[34:-16].unsqueeze(0)

    def save_observation(self, observation):
        data = Image.fromarray(observation)
        data.save('observation.png')
