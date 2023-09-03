import gymnasium as gym
from PIL import Image
import torch
import torchvision.transforms as transforms
from .Agent import Agent
from .Buffer import Buffer, EnvStateActionPairs

eps = 1e-6

class CarRacing:
    def __init__(self, size=40):
        self.size = size
        self.env = gym.make("CarRacing-v2", continuous=False)
        self.action_size = self.env.action_space.n
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(),
            transforms.ConvertImageDtype(torch.float),
        ])

    def reset(self) -> torch.Tensor:
        observation, _info = self.env.reset()
        return self._get_torch_tensor(observation)

    def agent_play(self, agent: Agent) -> Buffer:
        agent.reset()
        observation, _ = self.env.reset()
        buffers = Buffer()
        rewards = []
        for _ in range(1_000):
            old_observation = self._get_torch_tensor(observation)
            action = agent.get_action(old_observation)

            observation, reward, terminated, _, _ = self.env.step(action)
            buffers.add(
                EnvStateActionPairs(
                    state=old_observation,
                    next_state= self._get_torch_tensor(observation),
                    action=action,
                    reward=None
                )
            )
            rewards.append(reward)
            if terminated:
                break
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for index in range(len(buffers.entries)):
            buffers.entries[index].reward = rewards[index]
        return buffers

    def eval_agent_play(self, agent):
        agent.reset()
        observation, _ = self.env.reset()
        for _ in range(1_000):
            old_observation = self._get_torch_tensor(observation)
            action, old_h = agent.action(
                old_observation
            )
            observation, _, terminated, _, _ = self.env.step(action)

            yield self._raw_get_torch_tensor(observation)

            if terminated:
                break

    def random_play(self):
        observation, _ = self.env.reset()
        action = None
        previous_observation = None
        reward = None
        buffers = Buffer()
        rewards = []
        for _ in range(1_000):
            if action is not None and previous_observation is not None:
                buffers.add(
                    EnvStateActionPairs(
                        state=self._get_torch_tensor(previous_observation),
                        next_state= self._get_torch_tensor(observation),
                        action=action,
                        reward=None
                    )
                )
            action = self.env.action_space.sample()
            previous_observation = observation
            observation, reward, terminated, _, _ = self.env.step(action)
            rewards.append(reward)
            if terminated:
                break
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for index in range(len(buffers.entries)):
            buffers.entries[index].reward = rewards[index]
        return buffers

    def _get_torch_tensor(self, observation):
        """
        # making it easier for the model to distinguish 
        observation[:, :][observation[:, :] == 87] = 0
        # Player 2
        observation[:, :][observation[:, :] == 147] = 255
        # Player 1
        observation[:, :][observation[:, :] == 148] = 255
        """
        tensor = torch.from_numpy(observation).float() / 255
        tensor = tensor.permute(2, 0, 1)
     #   print(tensor.shape)
        # .permute(2, 0, 1) / 255
        # return tensor[:, 34:-16]
#        return tensor[34:-16].unsqueeze(0)
        return self.transforms(tensor)#.unsqueeze(0)

    def _raw_get_torch_tensor(self, observation):
        tensor = torch.from_numpy(observation).float() / 255
        return tensor

    def save_observation(self, observation):
        data = Image.fromarray(observation)
        data.save('observation.png')
