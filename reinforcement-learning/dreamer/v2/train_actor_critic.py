"""
Train the critic for the model
"""
import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class HiddenLatentRepresentation(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size) -> None:
        super().__init__()
        self.combined_size = latent_size + hidden_size

        self.latent = torch.nn.Sequential(*[
            nn.Linear(self.combined_size, output_size)
        ])

    def forward(self, latent, hidden):
        if len(hidden.shape) == 3:
            # hidden has batch on the second argument
            assert latent.shape[0] == hidden.shape[1], "Mismatch between batch"
            hidden = hidden.permute(1, 2, 0).reshape(latent.shape[0], -1)        
        combined = torch.concat((
            latent,
            hidden
        ), dim=1)
        assert combined.shape[1] == self.combined_size
        return self.latent(combined)

class RewardModel(nn.Module):
    def __init__(self, latent_size, hidden_size) -> None:
        super().__init__()

        self.reward_model = HiddenLatentRepresentation(latent_size, hidden_size, 1)
        self.discount_model = HiddenLatentRepresentation(latent_size, hidden_size, 1)
        # ^ core component.

    def forward(self, latent, hidden):
        return (
            self.reward_model.forward(latent, hidden),
            self.discount_model.forward(latent, hidden)
        )

class ActorCriticModel(nn.Module):
    def __init__(self, latent_size, hidden_size, actions) -> None:
        super().__init__()

        self.action_distribution = HiddenLatentRepresentation(latent_size, hidden_size, actions)
        self.critic = HiddenLatentRepresentation(latent_size, 1, 1)
        # parameters
        self.model_parameters = {
            "lambda": 0.95,
            "imagination_horizon": 0,
        }

    def get_actions(self, latent, hidden):
        return self.action_distribution.forward(latent, hidden)

    def get_critic(self, latent, hidden):
        pass

    def get_critic_value(self, imagination_model):
        return self.critic(imagination_model, torch.zeros((imagination_model.shape[0], 1)))

    def get_critic_value_imagination(self):
        pass
