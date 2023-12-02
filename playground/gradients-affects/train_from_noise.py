# based off https://www.cs.toronto.edu/~lczhang/321/lec/input_notes.html
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import torch.nn as nn
import torch
from optimization_playground_shared.plot.Plot import Plot, Image
import numpy as np

model = BasicConvModel()
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)
train, test = get_dataloader(subset=10_000)
for _ in range(1):
    (loss, acc) = iterator.train(train)

image = torch.randn(1, 1, 28, 28) + 0.5
image = torch.clamp(image, 0, 1)

started_noise = image.clone()

image.requires_grad = True
target_label = 8

optimizer = optim.Adam([image], lr=0.005)
criterion = nn.CrossEntropyLoss()

for _ in range(25_000):
    out = model(torch.clamp(image, 0, 1))
    loss = criterion(out, torch.Tensor([target_label]).long())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

plot = Plot()
plot.plot_image(
    images=[
        Image(
            title=f"Started noise",
            image=started_noise.squeeze()
        ),
        Image(
            title=f"From noise -> to {target_label}",
            image=image.squeeze()
        ),
    ],
    name='image_from_noise.png'
)
