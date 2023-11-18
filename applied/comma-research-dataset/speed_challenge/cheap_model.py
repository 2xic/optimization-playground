"""
Cheap model - this model is cheap and probably won't work very well
"""
from shared.dataloader import CarDriveDatasetWithDeltaFrame
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.models.BasicConvModel import get_output_shape
import matplotlib.style as mplstyle
import torch.nn.functional as F
import time
mplstyle.use('fast')

metrics_tracker = Tracker("speed_challenge")

class BasicConvModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.m = nn.BatchNorm2d(input_shape[0])
        self.conv1 = nn.Conv2d(input_shape[0], 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.output_shape = ((
            get_output_shape(input_shape,
            [self.conv1, self.pool, self.conv2, self.pool])
        ))
        self.fc1 = nn.Linear(16 * self.output_shape[0] * self.output_shape[1], 256)
        self.out = nn.Sequential(
            self.fc1,
            nn.Dropout(p=0.1),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.m(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

dataset = CarDriveDatasetWithDeltaFrame(
    fraction_of_dataset=1
)
model = BasicConvModel(input_shape=(3, 480, 640))
optimizer = optim.Adam(
    model.parameters(),
    weight_decay=0.01,
)
iterator = TrainingLoop(model, optimizer, loss=torch.nn.MSELoss())

loader = DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(10):
    start = time.time()
    (sum_loss, accuracy) = iterator.train(tqdm(loader, desc="Training"))
    print("Training time", time.time() - start)

    test_dataset = CarDriveDatasetWithDeltaFrame(
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Maybe the model just overfits after n epochs ? 
    predicted_y = [0]
    true_y = [0]
    start = time.time()
    for (X, y) in test_loader:
        for i in sum(model(X.to(iterator.device)).tolist(), []):
            predicted_y.append( i + predicted_y[-1]) 
        for i in sum(y.tolist(), []):
            true_y.append( i + true_y[-1])
        
    print("Model time", time.time() - start)
    start = time.time()
    plt.plot(predicted_y, label="Predicted")
    plt.plot(true_y, label="Truth")
    plt.legend(loc="upper left")
    plt.savefig(f'predictions.png')
    plt.clf()
    print("Plot time", time.time() - start)

    metrics_tracker.log(
        Metrics(
            epoch=epoch,
            loss=sum_loss,
            training_accuracy=accuracy, # accuracy in this sense is a bit inaccurate though
            prediction=Prediction.image_prediction(
                'predictions.png'
            )
        )
    )
