"""
They color quantize with kmeans

Modifications from their util code https://github.com/openai/image-gpt/blob/c6af2ebf57e2460c71fefa53cd9054b060cf716d/src/utils.py
"""
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from sklearn.cluster import KMeans
import numpy as np
import os
import torch

image_size = 20
train_dataloader, _ = get_dataloader(
    batch_size=32,
    transforms=Compose([
        Grayscale(),
        Resize((image_size, image_size)),
        ToTensor(),
    ])
)
n_clusters = 16

def iter_numpy():
    numpy_stacks = []
    channels = 1
    for index, (i, _) in enumerate(train_dataloader.dataset):
        numpy_stacks.append(i.squeeze(0).numpy().reshape(1, -1))
        if 1_0000 < index:
            break
    return np.stack(numpy_stacks).reshape(-1, channels)
    
def train():
    data = iter_numpy()
    model = KMeans(n_clusters=n_clusters).fit(data)
    np.save("centroids.npy", model.cluster_centers_)
    return torch.from_numpy(model.cluster_centers_)

def quantize(batch, centroids):
    def squared_euclidean_distance(a, b):
        b = torch.transpose(b, 0, 1)
        a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
        b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
        ab = torch.matmul(a, b)
        d = a2 - 2 * ab + b2
        return d
    b, c, h, w = batch.shape
    # [B, C, H, W] => [B, H, W, C]
    x = batch.permute(0, 2, 3, 1).contiguous()
    x = x.view(-1, c)
    quantized = torch.argmin((squared_euclidean_distance(
        x,
        centroids
    )), 1)
    return quantized.view(b, h, w)

def to_sequence(x):
    x = x.view(x.shape[0], -1)
    x = x.transpose(0, 1).contiguous()
    return x

def get_centroids_file():
    if os.path.isfile("centroids.npy"):
        centroids = torch.from_numpy(np.load("centroids.npy"))
        return centroids
    else:
        return train()

if __name__ == "__main__":
    first_row = train_dataloader.dataset[0]
    centroids = get_centroids_file()
    image = first_row[0].unsqueeze(0)
    print(quantize(image, centroids).shape)




