import torch
from model import Model
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from helper import sample_z, generate_mask
from torch.utils.data import DataLoader

"""
x -> 
"""


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_ds = MNIST("./", train=True, download=True,
                 transform=transforms.ToTensor())
X = torch.zeros((28, 28))
model = Model(device).to(device)
optimizer = torch.optim.Adam(model.parameters())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

#print(model)
#exit(0)
sample = sample_z(device=device)

for epoch in range(100):
    for i, (X, y) in enumerate(train_loader): # range(10_000):
        # as mentinoned in section 5.6
        #X,_ = train_ds[i]
        X = X.to(device)
        X = X.reshape((X.shape[0], 28 * 28))
        X += torch.rand((X.shape[0], 28 * 28), device=device) / 256
        X = torch.clamp(X, 0, 1)

        likelihood = model.forward(X)
        loss = -likelihood.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(epoch, loss.item())    

    x = model.backward(sample).cpu()
    plt.clf()
    plt.imshow(x.detach().numpy().reshape((28, 28)), cmap='gray')
    plt.savefig(f'results/{epoch}.png')

#plt.show()
