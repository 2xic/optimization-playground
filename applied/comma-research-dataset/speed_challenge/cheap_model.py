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
from optimization_playground_shared.plot.Plot import Plot, Figure

mplstyle.use('fast')


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
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
     #       nn.Tanh(),
        )

    def forward(self, x):
        x = self.m(x)
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x


def get_predicted_plots(
    model,
    dataloader,    
    device
):
    # Maybe the model just overfits after n epochs ? 
    predicted_y = [0]
    true_y = [0]
    start = time.time()
    with torch.no_grad():
        for (X, y) in dataloader:
            for i in sum(model(X.to(device)).tolist(), []):
                predicted_y.append( i + predicted_y[-1]) 
            for i in sum(y.tolist(), []):
                true_y.append( i + true_y[-1])
    return predicted_y, true_y

def train():
    num_workers = 4

    for method in ["optical_flow", "plain", "background_subtraction"]:
        metrics_tracker = Tracker(f"speed_challenge_{method}")
        dataset = CarDriveDatasetWithDeltaFrame(
            # reduce the amount of data we train to speed up the testing
            fraction_of_dataset=1,
            method=method,
        )
        # TODO: try resnet
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = BasicConvModel(input_shape=(3, 480, 640))
        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=0.01,
        )
        iterator = TrainingLoop(model, optimizer, loss=torch.nn.MSELoss())
        # More workers = more speed
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

        test_dataset = CarDriveDatasetWithDeltaFrame(
            train=False
        )
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=num_workers)
        train_loader_no_shuffle = DataLoader(dataset, batch_size=64, num_workers=num_workers)

        for epoch in range(1_00):
            start = time.time()
            (sum_loss, accuracy) = iterator.train(tqdm(loader, desc="Training"))
            print("Training time", time.time() - start)

            start = time.time()
            predicted_y, true_y = get_predicted_plots(
                model,
                test_loader,
                iterator.device
            )
            train_predicted_y, train_true_y = get_predicted_plots(
                model,
                train_loader_no_shuffle,
                iterator.device
            )
                
            print("Model time", time.time() - start)

            plot = Plot()
            plot.plot_figures(
                figures=[
                    Figure(
                        plots={
                            "predicted": predicted_y,
                            "truth": true_y,
                        },
                        title="Test dataset",
                        x_axes_text="Timestamp",
                        y_axes_text="Speed",
                    ),
                    Figure(
                        plots={
                            "predicted": train_predicted_y,
                            "truth": train_true_y,
                        },
                        title="Train dataset",
                        x_axes_text="Timestamp",
                        y_axes_text="Speed",
                    ),
                ],
                name='predictions.png'
            )
            
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

if __name__ == "__main__":
    train()
