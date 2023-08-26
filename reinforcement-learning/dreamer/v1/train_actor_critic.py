"""
Train the critic for the model
"""
import torch
import torch.nn as nn
from .config import Config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MlpRepresentation(nn.Module):
    def __init__(self, latent_size, output_size) -> None:
        super().__init__()
        self.combined_size = latent_size

        self.latent = torch.nn.Sequential(*[
            nn.Linear(self.combined_size, output_size),
            nn.Sigmoid(),
        ])

    def forward(self, latent):
        return self.latent(
            latent
        )

class RewardModel(nn.Module):
    def __init__(self, latent_size, hidden_size) -> None:
        super().__init__()

        self.reward_model = MlpRepresentation(latent_size, 1)
        self.discount_model = MlpRepresentation(latent_size, 1)
        # ^ core component.

    def forward(self, latent, hidden):
        return (
            self.reward_model.forward(latent, hidden),
            self.discount_model.forward(latent, hidden)
        )


class ActorCriticModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.action_distribution = MlpRepresentation(config.z_size, 1).to(config.device)
        self.critic = MlpRepresentation(config.z_size, 1).to(config.device)
        # parameters
        self.model_parameters = {
            "lambda": 0.95,
            "imagination_horizon": 1,
        }

    def get_actions(self, latent, hidden):
        return self.action_distribution.forward(latent, hidden)

    def get_critic(self, latent, hidden):
        pass

    def get_critic_value(self, imagination_model):
        return self.critic(imagination_model, torch.zeros((imagination_model.shape[0], 1)))

    def get_critic_value_imagination(self):
        pass
