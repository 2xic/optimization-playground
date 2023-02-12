from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader

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
    rows = 0
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
            rows += X.shape[0]
    return acc / rows
