from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn as nn
from optimization_utils.channel.Sender import send
from torchvision.datasets import MNIST
from torchvision import transforms

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 63)
        self.fc4 = nn.Linear(63, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        x = F.log_softmax(x, dim=1)
        #x = F.softmax(x, dim=1)
        return x



#train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
#X, y = train_ds[0]

#model = Net()
#model.load_state_dict(torch.load("model.pkt"))
#model.eval()
