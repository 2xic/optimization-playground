import torch
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
from torch import optim
from tqdm import tqdm
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from torchvision.utils import save_image

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Vae:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            input_shape=(1, 28, 28)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.vae.parameters())

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

    def loss(self, observation):
        self.vae.zero_grad()

        observation = observation.to(DEVICE)
        out = self.encode(observation)
        out = self.decode(out)
        out = out[:, :, :observation.shape[2], :observation.shape[3]]
     #   print(out.shape)
     #   print(observation.shape)
        loss = torch.nn.MSELoss(reduction='sum')(out, observation)
        loss.backward()

        self.optimizer.step()

        return loss, out

def train():
    progress = tqdm(range(1_000), desc='Training VAE')
    train, _ = get_dataloader(
        subset=1_00,
    )
    model = Vae()
    for epoch in progress:
        avg_loss = 0
        for (X, _) in train:
            (loss, out) = model.loss(X)
            avg_loss += loss
        if epoch % 10 == 0:
            save_image(X[:4], 'truth.png')
            save_image(out[:4], 'vae.png')
        progress.set_description(f'Loss {avg_loss.item()}')

if __name__ == "__main__":
    train()
