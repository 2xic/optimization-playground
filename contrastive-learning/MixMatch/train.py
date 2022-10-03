from itertools import combinations
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
optimizer = torch.optim.Adam(model.parameters())
dataloader = DataLoader(
    Cifar10Dataloader(),
    batch_size=1,
    shuffle=False,
    num_workers=0,
 )

for epoch in range(100):
    Xlabel = [] 
    AugmentedLabel = []
    with torch.no_grad():
        for index, (x, y, unlabeled) in enumerate(dataloader):
            x_augmentation = augmentations[random.randint(0, len(augmentations) - 1)](x).float()
            unlabeled_augmentations = [
                aug(unlabeled).float() for aug in augmentations
            ]
            y_hot = torch.zeros(10)
            y_hot[y] = 1
            Xlabel.append([
                x_augmentation, y_hot
            ])
            AugmentedLabel += [
                [aug, sharpen(model(aug), T=T)] for aug in unlabeled_augmentations
            ]

            if index > 10:
                break

    combined = list(Xlabel + AugmentedLabel)
    random.shuffle(combined)

    X_mixed = [
        MixUp()(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in zip(Xlabel, combined[:len(Xlabel)])     
    ]
    u_mixed = [
        MixUp()(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in zip(AugmentedLabel, combined[len(Xlabel):])
    ]

    loss_labeled = torch.tensor(0.0, dtype=torch.float)
    for (x, y) in (X_mixed):
        x_predicted = model(x)
        y_max = torch.argmax(y).reshape((1))
        loss_labeled += torch.nn.CrossEntropyLoss()(x_predicted, y_max)

    loss_unlabeled = torch.tensor(0.0, dtype=torch.float)
    for (x, y) in (u_mixed):
        x_predicted = model(x)
        loss_unlabeled += torch.nn.MSELoss()(x_predicted, y)

    optimizer.zero_grad()
    loss = loss_labeled + loss_unlabeled * lambda_value
    loss.backward()
    print(loss.item())
    optimizer.step()
    