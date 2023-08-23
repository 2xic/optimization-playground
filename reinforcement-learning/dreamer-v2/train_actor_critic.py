"""
Train the critic for the model
"""
import torch
import torch.nn as nn


class HiddenLatentRepresentation(nn.Module):
    def __init__(self, latent_size, hidden_size) -> None:
        super().__init__()
        self.combined_size = latent_size + hidden_size

        self.latent = torch.nn.Sequential(*[
            nn.Linear(self.combined_size, 128)
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

class ActorCriticModel(nn.Module):
    def __init__(self, latent_size, hidden_size) -> None:
        super().__init__()

        self.reward_model = HiddenLatentRepresentation(latent_size, hidden_size)
        self.discount_model = HiddenLatentRepresentation(latent_size, hidden_size)
        # ^ core component.

    def forward(self, latent, hidden):
        return (
            self.reward_model.forward(latent, hidden),
            self.discount_model.forward(latent, hidden)
        )
