import torch
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
from torch import optim
from tqdm import tqdm
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.plot.Plot import Plot, Image

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
        loss = torch.nn.functional.mse_loss(out, observation)
        loss.backward()

        self.optimizer.step()

        return loss, out

def forward_gaussian(model, X, std, mean=0):
    vae_encode = model.encode(X)[:1]
    gaussian_noise = torch.empty(vae_encode.shape).normal_(mean=mean, std=std)
    decoded = model.decode(vae_encode * gaussian_noise)
    return decoded

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
            plot = Plot()

            decoded_std_1 = forward_gaussian(model, X, std=1)
            decoded_std_0_5 = forward_gaussian(model, X, std=0.5, mean=0.5)
            decoded_std_0_1 = forward_gaussian(model, X, std=0, mean=0.1)
            plot.plot_image(
                images=[
                    Image(
                        title="Real",
                        image=X[:1].squeeze(0),
                    ),
                    Image(
                        title="Vae",
                        image=out[:1].squeeze(0),
                    ),
                    Image(
                        title="Vae + Gaussian N(mean=0, std=1)",
                        image=decoded_std_1[:1].squeeze(0),
                    ),
                    Image(
                        title="Vae + Gaussian N(mean=0.5, std=0.5)",
                        image=decoded_std_0_5[:1].squeeze(0),
                    ),
                    Image(
                        title="Vae + Gaussian N(mean=0, std=0.1)",
                        image=decoded_std_0_1[:1].squeeze(0),
                    )
                ],
                name='example.png'
            )
        progress.set_description(f'Loss {avg_loss.item()}')

if __name__ == "__main__":
    train()
