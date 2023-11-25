"""
https://arxiv.org/pdf/1608.01230.pdf

1. First we train an encoder
2. We train a gan with the encoder
3. We win.

Transition model
-> they train the encoder minus the encoder output of next frame

"""
import torch
from .vae import SimpleVaeModel
from torch import optim
from tqdm import tqdm
from shared.dataloader import CarDriveDataset
from torch.utils.data import DataLoader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.plot.Plot import Plot, Image

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Vae:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            # 3, 80, 160 is same as the authors used as size
            input_shape=(3, 80, 160),
            z_size=1024,
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.vae.parameters())

    # I thought that maybe removing the randomness would help, but it makes things less smooth
    """
    def encode(self, observation):
        observation = observation.to(DEVICE)
        (mean, log_var) = self.vae.encode(observation)
        var = torch.exp(0.5 * log_var)
        z = mean + var 
        return z
    """

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
    
    def _forward_and_backward(self, observation):
        out = self.encode(observation)
        out = self.decode(out)
        assert out.shape == observation.shape, out.shape
        return out

    def loss(self, observation):
        self.vae.zero_grad()

        observation = observation.to(DEVICE)
        out = self.encode(observation)
        out = self.decode(out)
        loss = torch.nn.functional.mse_loss(out, observation)
        loss.backward()

        self.optimizer.step()

        return loss, out

def train():
    from torchvision.transforms import v2
    dataset = CarDriveDataset(
        fraction_of_dataset=1,
        transformers=v2.Compose([
            v2.RandomResizedCrop(size=(80, 160), antialias=True)
        ])
    )
    model = Vae()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    metrics_tracker = Tracker("learning_a_driving_simulator_encoder")

    for epoch in range(1_000):
        sum_loss = 0
        progress = tqdm(loader)
        for _, (X, _) in enumerate(progress):
            X = X.float() / torch.max(X)
            (loss, out) = model.loss(X)
            progress.set_description(f'Loss {loss.item()}')
            sum_loss += loss

        inference = Plot().plot_image([
            Image(
                image=X[0],
                title='Real'
            ),
            Image(
                image=out[0],
                title='Encoder -> Decoder'
            )
        ], f'inference.png')
    
        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=sum_loss,
                training_accuracy=None,
                prediction=Prediction.image_prediction(
                    inference
                )
            )
        )

if __name__ == "__main__":
    model = Vae()
#    print(model._forward_and_backward(torch.zeros((1, 3, 80, 160))))
    train()
