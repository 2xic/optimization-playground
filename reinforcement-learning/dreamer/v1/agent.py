from .train_actor_critic import ActorCriticModel
from .train_world_model import WorldModel
from .config import Config
import torch
from dataclasses import dataclass

@dataclass
class ImaginedTrajectories:
    state: torch.Tensor
    action: torch.Tensor


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.world_model = WorldModel(
            config
        ).to(config.device)
        self.actor_critic = ActorCriticModel(
            config
        )
        self.reset()

    def reset(self):
        self.last_state = torch.zeros((1, self.config.z_size), device=self.config.device)
        self.last_action = torch.zeros((1, 1), device=self.config.device)

    def forward(self, observation, last_state, action):
        combined_tensor = torch.concat((
            last_state, 
            action
        ), dim=1)
        (encoded, logvar, mu) = self.world_model.representation(
            observation,
            combined_tensor
        )
        reward = self.world_model.reward_model(encoded)
        return (
            (encoded, logvar, mu),
            reward
        )

    # Should be called from the env only
    def get_action(self, observation):
        observation = observation.to(self.config.device).float()
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        self.last_state, _, _ = self.forward(observation, self.last_state, self.last_action)[0]
        self.last_action = self.actor_critic.action_distribution(self.last_state)
       # print(int((self.last_action * (self.config.action_size - 1)).item()))
        return int((self.last_action * (self.config.action_size - 1)).item())

    def _get_action(self, observation, last_state, action):
        (encoded, _) = self.forward(observation, last_state, action)
        actions = self.actor_critic.action_distribution(encoded)
        return ((actions * self.config.action_size).item())

    """
    imagine trajectory
    """
    def transition(self, encoded, action):
        encoded = self.world_model.transition(
            encoded,
            action
        )
        reward = self.world_model.reward_model(encoded)
        return (
            encoded,
            reward
        )
    def rollout(self, observation, last_state, action):
        rollout_steps = 5
        (encoded, _) = self.forward(observation, last_state, action)
        actions = self.actor_critic.action_distribution(encoded)
        output = []
        for _ in range(rollout_steps):
            encoded = self.world_model.transition(
                encoded,
                torch.argmax(actions, dim=1).reshape(encoded.shape[0], -1)
            )
            actions = self.actor_critic.action_distribution(encoded)
            output.append(ImaginedTrajectories(
                state=encoded,
                action=actions
            ))
        return output
