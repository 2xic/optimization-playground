from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from model import Net
import torch

train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
X, y = train_ds[0]

for trained in [True, False]:
    plt.clf()

    name = "trained" if trained else "untrained"

    model = Net()
    if trained:
        model.load_state_dict(torch.load("model.pkt"))
    model.eval()

    layers = list(model.modules())[0]
    conv1: torch.nn.Conv2d = layers.conv1


    conv1_weight = conv1.weight.detach().numpy()
    f, ax_arr = plt.subplots(2,6)
    f.suptitle(name)

    for i in range(conv1_weight.shape[0]):
        image = conv1_weight[i][0]
        ax_arr[i // 6, i % 6].imshow(image,  cmap='Greys')
        ax_arr[i // 6, i % 6].axis('off')
    plt.savefig(f'{name}_conv1.png')
