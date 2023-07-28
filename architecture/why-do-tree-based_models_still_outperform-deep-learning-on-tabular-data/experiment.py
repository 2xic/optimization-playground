import numpy as np
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.plot.Plot import Plot, Scatter, ScatterEntry
import torch.nn as nn

"""
Simple experiment with "fake" tabluar data

f(x, y) = |cos(2 * x + y)|

We have the features

x | y | f(x, y) | random number | random number | random number | random number 

"""
class RandomTabularData:
    def __init__(self, random_features) -> None:
        self.samples_count = 256
        self.function = lambda x, y: np.abs(np.cos(2 * x + y))
        self.samples = [
            (np.abs(np.sin(i)), np.abs(np.cos(i)), self.function(np.abs(np.sin(i)), np.abs(np.cos(i))))
            for i in range(self.samples_count)
        ]
        self.random_features = random_features

        self.TRAINING_SPLIT = 0.66
        self.SPLIT_INDEX = int(self.TRAINING_SPLIT * self.samples_count)
        self.X = torch.rand((self.samples_count, 2 + self.random_features))
        self.Y = torch.rand((self.samples_count, 1))
        for index, (x, y, z) in enumerate(self.samples):
            self.X[index][0] = x
            self.X[index][1] = y
            self.Y[index][0] = z
    
    def get_training_dataset(self):
        return self.X[:self.SPLIT_INDEX], self.Y[:self.SPLIT_INDEX]

    def get_test_dataset(self):
        return self.X[self.SPLIT_INDEX:], self.Y[self.SPLIT_INDEX:]

    
def fit_regressor(X, y, X_test, y_test):
    model = XGBRegressor()
    model.fit(X, y)

    return mean_squared_error(y_test, model.predict(X_test))

def train_nn(X, y, X_test, y_test):
    train_dataloader = get_dataloader(X, y)
    test_dataloader = get_dataloader(X_test, y_test)

    model = nn.Sequential(*[
        nn.Linear(X.shape[-1], 256),
        nn.Sigmoid(),
        nn.Linear(256, 512),
        nn.Sigmoid(),
        nn.Linear(512, 256),
        nn.Sigmoid(),
        nn.Linear(256, y.shape[-1]),
        nn.Tanh(),
    ])
    optimizer = torch.optim.Adam(model.parameters())
    
    training_loop = TrainingLoop(model, optimizer, loss=nn.MSELoss())
    for _ in range(100):
        training_loop.train(train_dataloader)
    return mean_squared_error(y_test, model(X_test).detach().numpy())


if __name__ == "__main__":
    """
    This looks pretty random to me, I guess we should also do some statistical test
    """
    xgboost_scatter_entry = ScatterEntry(
        [],
        []
    )
    nn_scatter_entry = ScatterEntry(
        [],
        []
    )
    for random_variables_count in range(0, 32):
        dataset = RandomTabularData(
            random_features=random_variables_count
        )
        X, y = dataset.get_training_dataset()
        X_test, y_test = dataset.get_test_dataset()
        

        xgboost_error = fit_regressor(X, y, X_test, y_test)
        nn_error = train_nn(X, y, X_test, y_test)

        xgboost_scatter_entry.X.append(random_variables_count)
        xgboost_scatter_entry.y.append(xgboost_error)

        nn_scatter_entry.X.append(random_variables_count)
        nn_scatter_entry.y.append(nn_error)

    scatter = Scatter(
        {
            "xgboost": xgboost_scatter_entry,
            "nn": nn_scatter_entry
        },
        "Testing on randomness dataset (MSE error)"
    )
    Plot().plot_scatter(
        scatter,
        "experiment.png"
    )
