import torch.optim as optim
import torch 
import torch.nn as nn
from vae import SimpleVaeModel
from train_actor_critic import HiddenLatentRepresentation

device = torch.device('cuda)') if torch.cuda.is_available() else torch.device('cpu')

"""
The core model deals with the memory
"""
class RecurrentModel:
    def __init__(self, latent_size, action_size) -> None:
        #latent_size = 32
        #action_size = 4
        self.input_size = latent_size + action_size
        # params
        self.hidden_Size = 32
        self.number_of_layers = 2
        self.recurrent = nn.GRU(self.input_size, self.hidden_Size, self.number_of_layers, batch_first=True)

    def initial_state(self, batch_size):
        return torch.randn(self.number_of_layers,batch_size,  self.hidden_Size).to(device)

    def forward(self, latent, action, hidden):
        assert len(latent.shape) == len(action.shape)
        assert (latent.shape[0]) == (action.shape[0])
        X = torch.concat((
            latent,
            action   
        ), dim=1)
        if len(X.shape) == 2:
            X = X.unsqueeze(1)
        assert len(X.shape) == 3
        if hidden is None:
            hidden = self.initial_state(X.shape[0])
        output, hidden = self.recurrent(X, hidden)
        output = output[:, -1, :]
        return output, hidden
"""
The representation model takes in the input image and crates a representation vector.

This also has the component for going from the representation vector back into a env like image.
"""
class WorldModel:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            input_shape=(1, 40, 40),
            conv_shape=[
                32,
                64,
                128,
            ],
            z_size=128,
        )
        self.transition_model = torch.nn.Sequential(*[
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
        ])
        self.image_predictor_model = HiddenLatentRepresentation(
            128,
            64
        )
        self.optimizer = optim.Adam(
            list(self.vae.parameters()) +
            list(self.transition_model.parameters())
        )

    def representation(self, observation, hidden):
        """
        goes from the image and hidden state into a latent vector
        """
        # hidden has batch on the second argument
        assert observation.shape[0] == hidden.shape[1], "Mismatch between batch"
        hidden = hidden.permute(1, 2, 0).reshape(observation.shape[0], -1)        

        observation = observation.to(device)
        (mean, log_var) = self.vae.encode(observation, hidden)
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def image_predictor(self, hidden, latent):
        """
        goes from latent vector and hidden state into a image
        """
        assert latent.shape[0] == hidden.shape[1], "Mismatch between batch"
        hidden = hidden.permute(1, 2, 0).reshape(latent.shape[0], -1)        
        return self.image_predictor_model(hidden, latent)

    def transition(self, hidden_state):
        return self.transition_model(hidden_state)
