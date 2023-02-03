from model import Net
from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from WeightSchedule import WeighSchedule
from predictor import Predictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataloader = DataLoader(
    Cifar10Dataloader(
        test=True
    ),
    batch_size=64,
    shuffle=True,
)

def eval_model(model):
    acc = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            acc += torch.sum(
                torch.argmax(
                    model(X),
                    1
                ) == y
            )
    return acc / len(dataloader)
