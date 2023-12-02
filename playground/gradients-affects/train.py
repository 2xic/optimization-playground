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


image, image_class  = next(iter(train))
image.requires_grad = True
target_label = 8
image = image[:1, :, : ,:]
image_class = image_class[:1]

criterion = nn.CrossEntropyLoss()

out = model(image)
loss = criterion(out, torch.Tensor([target_label]).long())
image_grad = torch.autograd.grad(loss, image, retain_graph=True)
image_grad = image_grad[0][0].numpy()
image_grad = (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min())
image_grad = np.concatenate([
    image_grad,
    image_grad,
    image_grad
], axis=0)
print(image_grad.shape)

plot = Plot()
plot.plot_image(
    images=[
        Image(
            title=f"{image_class.item()} -> {target_label}",
            image=image_grad
        ),
    ],
    name='grad.png'
)
