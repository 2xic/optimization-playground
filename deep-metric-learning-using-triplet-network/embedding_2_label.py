from collections import defaultdict
from model import SimpleModel
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import random
from triplet_model import TripletModel
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch

class SuperSimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


"""
if the emending is storing useful information - 
you should be able to learn labels from it
"""
train_ds = MNIST("./", train=True, download=True,
                 transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)
model_state_dict = torch.load('model_state')['model_state_dict']

model = TripletModel()
model.load_state_dict(model_state_dict)
model.eval()

new_model = SuperSimpleModel()
optimizer = torch.optim.Adam(new_model.parameters())


for epoch in range(10):
    for batch, (X, y) in enumerate(train_loader):
        with torch.no_grad():
            x_embedded = model.embedded(X)
        
        loss = torch.nn.CrossEntropyLoss()(
            new_model(x_embedded),
            y
        )
        new_model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 250 == 0:
            print(loss)
    print(f"epoch : {epoch}")
    
test_ds = MNIST("./", train=False, download=True,
                 transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=32)

acc = 0
batch = 0
for (X, y) in test_loader:
    y_dot = torch.argmax(new_model(model.embedded(X)), dim=1)
    #print((y == y_dot).long())
    acc += ((y == y_dot).long().sum())
    batch += y.shape[0]

"""
tested this with 8 output
accuracy around 10%

tested this with 16 output
accuracy around 25%

retrained with tanh, and output drops to 0.1135

full image size is 2352
"""
print(acc / batch)
