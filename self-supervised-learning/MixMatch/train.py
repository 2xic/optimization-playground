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
device = 'cpu' # 'cuda:0'
optimizer = torch.optim.Adam(model.parameters())
batch_size = 64
dataloader = DataLoader(
    Cifar10Dataloader(),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

model.to(device)
step_size = 4

for epoch in range(1_00):
    for index, (x, y, unlabeled) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        unlabeled = unlabeled.to(device)

        x_augmentation = augmentations(x).float()
        #X_label += list(zip(x_augmentation, y))
        """
        TODO: This could be more tensor effective
        """
        aug = augmentations(unlabeled)
        with torch.no_grad():
            aug_label = sharpen(model(aug), T=T)
        
        split = batch_size // 2

        (X_mixed, y_mixed) = MixUp()(x_augmentation[:split], y[:split], aug[:split], aug_label[:split], device)
        (u_mixed, uy_mixed) = MixUp()(x_augmentation[split:], y[split:], aug[split:], aug_label[split:], device)

        loss_labeled = torch.tensor(0.0, dtype=torch.float, device=device)
        x_predicted = model(X_mixed)
        y_max = torch.argmax(y_mixed, dim=1).reshape((y_mixed.shape[0]))
        loss_labeled += torch.nn.CrossEntropyLoss()(x_predicted, y_max)

        x_predicted = model(u_mixed)
        loss_unlabeled = torch.nn.MSELoss()(x_predicted, uy_mixed)

        optimizer.zero_grad()
        loss = loss_labeled + loss_unlabeled * lambda_value
        loss.backward()

        if index % 10 == 0:
            with torch.no_grad():
                predicted_y = torch.argmax(model(x), dim=1)
                y_max = torch.argmax(y, dim=1)
                equal =  predicted_y == y_max
                #print(predicted_y.shape)
                #print(y_max.shape)
                #print(predicted_y)
                #print(y_max)
                #print(equal)
                accuracy = ((equal.sum()) / batch_size).item()
                with open("train_accuracy.txt", "a") as file:
                    file.write(f"{accuracy}\n")
                print(epoch, index, loss.item(), accuracy)
        optimizer.step()
