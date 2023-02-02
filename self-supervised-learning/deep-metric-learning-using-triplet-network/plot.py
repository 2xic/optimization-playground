from triplet_model import TripletModel
import numpy as np
from sklearn.manifold import TSNE
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

train_ds = MNIST("./", train=True, download=True,
                 transform=transforms.ToTensor())

train_data = ([train_ds[index] for index in range(32)])

X = [train_data[i][0] for i in range(len(train_data))]
y = [train_data[i][1] for i in range(len(train_data))]


model_state_dict = torch.load('model_state')['model_state_dict']

model = TripletModel()
model.load_state_dict(model_state_dict)


latent_x = [
    model.embedded(i).tolist()[0] for i in X
]

X = np.array(latent_x)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', 
                  n_iter=10_000,
                  perplexity=3).fit_transform(X)

# list form from://matplotlib.org/stable/tutorials/colors/colormaps.html
colors = [
    'aliceblue',
    'antiquewhite',
    'aqua',
    'aquamarine',
    'azure',
    'beige',
    'bisque',
    'black',
    'blanchedalmond',
    'lightsteelblue',
    'lightyellow',
    'lime',
    'navajowhite',
    'navy',
    'oldlace',
    'olive',
    'olivedrab',
    'orange',
    'orangered',
    'orchid',
    'palegoldenrod',
    'palegreen',
    'paleturquoise',
    'palevioletred',
    'papayawhip',
    'peachpuff',
    'peru',
    'pink',
]

#print(X_embedded)
for x, y in zip(X_embedded, y):
    output = plt.scatter(x[0], x[1], color=colors[y])

plt.legend(loc='upper left')
plt.show()
