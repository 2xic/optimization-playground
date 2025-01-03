from model import Vae
from dataloader import Dataset
from coco_image_loader import CocoImageLoader
from torch.utils.data import DataLoader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.plot.Plot import Plot, Image
import torch
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
import torchvision
metrics_tracker = Tracker("sharpen_image")

def train():
    model = Vae()
    image_loader = CocoImageLoader()
    dataloader = None
    if False:
        dataset = Dataset(image_loader, max_items=1_000)
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
        )
    else:
        dataloader, _ = get_dataloader(
            subset=1_00,
        )

    for epoch in range(1_000):
        #for (X, y) in dataloader:
        for (X, _) in dataloader:
            y = X.clone()
            X = torchvision.transforms.Compose([
                torchvision.transforms.GaussianBlur(kernel_size=3),
            ])(X)
            loss, sharpen_y = model.loss(X, y)
            print(f"loss {loss}")
            inference = Plot().plot_image([
                Image(
                    image=X[0].detach().to(torch.device('cpu')).numpy(),
                    title='input'
                ),
                Image(
                    image=y[0].detach().to(torch.device('cpu')).numpy(),
                    title='truth'
                ),
                Image(
                    image=sharpen_y[0].detach().to(torch.device('cpu')).numpy(),
                    title='model sharped'
                ),
            ], f'inference.png')
            metric = Metrics(
                epoch=epoch,
                loss=loss.item(),
                training_accuracy=None,
                prediction=Prediction.image_prediction(
                    inference
                )
            )
            metrics_tracker._log(metric)

if __name__ == "__main__":
    train()
