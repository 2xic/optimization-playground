import torch.optim as optim
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch
from config import IMAGE_SIZE

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Vae:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            input_shape=(1, IMAGE_SIZE, IMAGE_SIZE),
            conv_shape=[
                16,
                32,
                64
            ],
            z_size=5024,
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)

    def encode(self, observation):
        observation = observation.to(DEVICE)
        (mean, log_var) = self.vae.encode(observation)
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var*epsilon
        return z

    def decode(self, observation):
        observation = observation.to(DEVICE)
        return self.vae.decode(observation)

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss(self, x, y ):
        self.vae.zero_grad()
        sharpen_y = self.forward(x.to(DEVICE))
        sharpen_y = sharpen_y[:, :, :x.shape[2], :x.shape[3]]

        validate = False
        if validate:
            assert torch.max(sharpen_y).item() <= 1, torch.max(sharpen_y).item()
            assert torch.max(y).item() <= 1, torch.max(y).item()
            assert torch.max(x).item() <= 1, torch.max(x).item()
            assert 0 <= torch.min(sharpen_y).item(), torch.min(sharpen_y).item()
            assert 0 <= torch.min(y).item(), torch.min(y).item()
            assert 0 <= torch.min(x).item(), torch.min(x).item()
        
        loss = torch.nn.MSELoss(reduction='sum')(sharpen_y, y.to(DEVICE))
        loss.backward()
        self.optimizer.step()

        return loss, sharpen_y
