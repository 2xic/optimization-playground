import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.plot.Plot import Plot, Image
import torchvision
from enum import IntEnum

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(
            28 * 28, 256
        )
        self.layer_2 = nn.Linear(
            256, 128
        )
        self.layer_3 = nn.Linear(
            128, 10
        )
        
    def forward(self, X):
        return self._forward(X)[0]

    def forward_phi(self, X):
        return self._forward(X)[1]

    def _forward(self, X):
        X = X.reshape(X.shape[0], 28 * 28)
        X = nn.ReLU()(self.layer_1(X))
        phi = self.layer_2(X)
        X = nn.ReLU()(phi)
        X = nn.LogSoftmax(dim=1)(self.layer_3(X))

        return (
            X,
            phi
        )


class Mode(IntEnum):
    RANDOM = 1
    EYE = 2

def get_activations(
        model: Model, 
        test: DataLoader,
        mode: Mode
    ):
    model.eval()
    image_count = 8

    plot_images = []

    name = 'random_vector.png' if mode == Mode.RANDOM else 'eye_vector.png'
    for index in range(3):
        title = f"High activation with random vector ({index})" if mode == Mode.RANDOM else f"High activation with eye vector ({index})"
    
        plot = Plot()  
        random_vector = torch.rand(128) if mode == Mode.RANDOM else torch.eye(128)[index,:]
        for images, _ in test:
            phi = model.forward_phi(images)
            values = torch.mv(phi, random_vector)
            
        top_img = images[torch.argsort(values.data)[-image_count:]]
        image = torchvision.utils.make_grid(top_img, normalize=True)
        moved = image.permute(1, 2, 0)
        plot_images.append(
            Image(
                image=moved,
                title=title
            )
        )

    plot.plot_image(
        plot_images,
        name,
        row_direction=False
    )

if __name__ == "__main__":
    (train, test) = get_dataloader()
    model = Model()
    optimizer = optim.Adam(model.parameters())
    loop = TrainingLoop(
        model,
        optimizer
    )
    for _ in range(1_00):
        (loss, acc) = loop.train(train)
        print(f"Loss {loss}, acc: {acc}")

    top_images = get_activations(
        model,
        test,
        Mode.RANDOM
    )
    top_images = get_activations(
        model,
        test,
        Mode.EYE
    )

