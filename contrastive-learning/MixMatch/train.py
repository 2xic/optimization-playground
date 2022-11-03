from model import Net
from Dataloader import Cifar10Dataloader
from torch.utils.data import DataLoader
import random
import torch
from sharpen import sharpen
from mixup import MixUp
from augmentations import augmentations
from hyperparameters import T, lambda_value
import torch
"""
1. Run augmentation on a batch
    - Add a test util to the optimization library maybe
2. Feed the model the augmentations
    - Get the average predictions.
    - Sharpen it
3. Create a batch W of labeled and unlabeled batch
    - MixUp labeled batch
    - MixUp unlabeled batch
4. Loss is applied to the Created batches weighted on lambda.
    - Cross entropy on labeled batch
    - Squared l2 loss on unlabeled batch
"""

model = Net()
device = 'cuda:0'
optimizer = torch.optim.Adam(model.parameters())
dataloader = DataLoader(
    Cifar10Dataloader(),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    )
model.to(device)
step_size = 32

for epoch in range(1_000):
    Xlabel = [] 
    AugmentedLabel = []
    accuracy = 0
    with torch.no_grad():
        for index, (x, y, unlabeled) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            unlabeled = unlabeled.to(device)

            x_augmentation = augmentations[random.randint(0, len(augmentations) - 1)](x).float()
            Xlabel.append([
                x_augmentation, y
            ])
            """
            TODO: This could be more tensor effective
            """
            unlabeled_augmentations = [
                aug(unlabeled).float() for aug in augmentations
            ]
            AugmentedLabel += [
                [aug, sharpen(model(aug), T=T)] for aug in unlabeled_augmentations
            ]

            if index > step_size:
                batch_size = x.shape[0]
                y_pred = model(x)
                y_pred = torch.argmax(y_pred.clone().detach(), dim=1)
                accuracy += torch.count_nonzero(y_pred - y)
                break

    combined = list(Xlabel + AugmentedLabel)
    random.shuffle(combined)

    """
    TODO: This could be more tensor effective
    """
    X_mixed = [
        MixUp()(x1, y1, x2, y2, device) for ((x1, y1), (x2, y2)) in zip(Xlabel, combined[:len(Xlabel)])     
    ]
    u_mixed = [
        MixUp()(x1, y1, x2, y2, device) for ((x1, y1), (x2, y2)) in zip(AugmentedLabel, combined[len(Xlabel):])
    ]

    loss_labeled = torch.tensor(0.0, dtype=torch.float, device=device)
    for (x, y) in (X_mixed):
        x_predicted = model(x)
        y_max = torch.argmax(y).reshape((1)).to(device)
        loss_labeled += torch.nn.CrossEntropyLoss()(x_predicted, y_max)

    loss_unlabeled = torch.tensor(0.0, dtype=torch.float, device=device)
    for (x, y) in (u_mixed):
        x_predicted = model(x)
        y = y.to(device)
        loss_unlabeled += torch.nn.MSELoss()(x_predicted, y)

    optimizer.zero_grad()
    loss = loss_labeled + loss_unlabeled * lambda_value
    loss.backward()
    if epoch % 10 == 0:
        with open("train_accuracy.txt", "a") as file:
            file.write(f"{accuracy / step_size}\n")
        print(loss.item())
    optimizer.step()
