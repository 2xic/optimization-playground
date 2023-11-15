"""
Cheap model - this model is cheap and probably won't work very well
"""
from dataloader import CarDriveDatasetWithDeltaFrame
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

dataset = CarDriveDatasetWithDeltaFrame(
    fraction_of_dataset=1
)
model = BasicConvModel(input_shape=(3, 480, 640), num_classes=1)
model.out = nn.Sequential(
    model.fc1,
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
    nn.Linear(64, 1)
)
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer, loss=torch.nn.MSELoss())

loader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(10):
    (loss, accuracy) = iterator.train(tqdm(loader, desc="Training"))
    print(loss, accuracy)

    test_dataset = CarDriveDatasetWithDeltaFrame(
        train=False,
        fraction_of_dataset=0.25
    )
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predicted_y = [0]
    true_y = [0]
    for (X, y) in test_loader:
        for i in sum(model(X.to(iterator.device)).tolist(), []):
            predicted_y.append( i + predicted_y[-1]) 
        for i in y.tolist():
            true_y.append( i + true_y[-1]) 

    plt.plot(predicted_y, label="Predicted")
    plt.plot(true_y, label="Truth")
    plt.legend(loc="upper left")
    plt.savefig(f'predictions_{epoch}.png')
    plt.clf()

    print(f'predictions_{epoch}.png')
    print("")

