"""
This does not contain the transformer part, just the cookbookl
"""

import torch
from torch import optim
from tqdm import tqdm
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embed_dim=64):
        super().__init__()

        # Encoder: (1, 28, 28) -> (embed_dim, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, padding=1),
        )

        # cookbook: K vectors of dimension D
        self.cookbook = nn.Embedding(num_embeddings, embed_dim)
        self.cookbook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # Decoder: (embed_dim, 7, 7) -> (1, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim

    def quantize(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        dist = torch.cdist(z_flat, self.cookbook.weight)
        indices = dist.argmin(dim=1)
        z_q_flat = self.cookbook(indices)

        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        loss = (z - z_q).pow(2).mean()

        z_q = z + (z_q - z).detach()

        return z_q, loss

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantize(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss, vq_loss

    def decode_indices(self, indices):
        B, H, W = indices.shape
        z_q = self.cookbook(indices.flatten())
        z_q = z_q.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return self.decoder(z_q)


class VQVaeWrapper:
    def __init__(self) -> None:
        self.vqvae = VQVAE().to(DEVICE)
        self.optimizer = optim.Adam(self.vqvae.parameters())

    def loss(self, observation):
        self.vqvae.zero_grad()

        observation = observation.to(DEVICE)
        (out, recon_loss, vq_loss) = self.vqvae(observation)

        loss = recon_loss + vq_loss
        loss.backward()

        self.optimizer.step()

        return loss, out


def plot_cookbook(model, num_entries=64):
    model.eval()

    cols = 8
    rows = num_entries // cols
    _, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

    with torch.no_grad():
        for i in range(num_entries):
            indices = torch.full((1, 7, 7), i, dtype=torch.long, device=DEVICE)
            img = model.decode_indices(indices)

            ax = axes[i // cols, i % cols]
            ax.imshow(img[0, 0].cpu(), cmap="gray")
            ax.set_title(f"{i}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("cookbook_entries.png")
    plt.show()


def train():
    progress = tqdm(range(1_000), desc="Training VAE")
    train, _ = get_dataloader(
        subset=1_00,
    )
    model = VQVaeWrapper()
    for epoch in progress:
        avg_loss = 0
        for X, _ in train:
            (loss, out) = model.loss(X)
            avg_loss += loss
        if epoch % 10 == 0:
            save_image(X[:4], "truth.png")
            save_image(out[:4], "vae.png")
            plot_cookbook(model.vqvae)
        progress.set_description(f"Loss {avg_loss.item()}")


if __name__ == "__main__":
    train()
