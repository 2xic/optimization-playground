import torch.optim as optim
import torch
import torch.nn as nn
from .train_actor_critic import MlpRepresentation
from .config import Config
from .vae import SimpleVaeModel

"""
The representation model takes in the input image and crates a representation vector.

This also has the component for going from the representation vector back into a env like image.
"""


class WorldModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.vae = SimpleVaeModel(
            config=config,
            input_shape=(1, config.image_size, config.image_size),
            conv_shape=[
                32,
                64,
                128,
                256,
            ],
            z_size=config.z_size,
        ).to(config.device)
    #    print(self.vae)
        self.transition_model = MlpRepresentation(
            latent_size=(config.z_size + 1),
            output_size=config.z_size
        ).to(config.device)
        self.reward_model = MlpRepresentation(
            latent_size=config.z_size,
            output_size=1
        ).to(config.device)

        self.optimizer = optim.Adam(
            list(self.vae.parameters())
        )

    # this should be possible to train fully in isolation
    def save_model(self):
        pass

    def representation(self, observation, combined):
        """
        goes from the image and hidden state into a latent vector
        """
        (mean, log_var) = self.vae.encode(observation, combined)
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(self.config.device)
        z = mean + var*epsilon
        return (z, log_var, mean)

    def transition(self, latent, action):
        combined = torch.concat(
            (latent, action),
            dim=1
        )
        return self.transition_model(combined)

    def decode_latent(self, latent):
        """
        goes from latent into the image of the environment
        """
        # hidden has batch on the second argument
        return self.vae.decode(latent)
