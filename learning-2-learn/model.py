from distutils.log import error
from typing import Any, Tuple
import typing
from dataset import get_quadratic_function_error
import pytorch_lightning as pl
import torch.nn as nn
import torch


class Learning2Learn(pl.LightningModule):
    def __init__(self,
                 n_features,
                 hidden_size,
                 num_layers,
                 dropout):
        super(Learning2Learn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: paper uses two lstm 
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, y, logging=False):
        step_size = 100
        t_0 = torch.rand((1, 1, 1))
        hidden = (
            torch.zeros((self.num_layers, 1, self.hidden_size)),
            torch.zeros((self.num_layers, 1, self.hidden_size))
        )
        error_function = get_quadratic_function_error(x, y)
        error_over_time = None
        y_pred = None
        x_history = [
            t_0
        ]
        for _ in range(step_size):
            func_error = error_function(x_history[-1]).reshape((1, 1, -1))
            lstm_out, hidden = self.lstm(func_error, hidden)
            y_pred = self.linear(lstm_out[:, -1])

            new_x = x_history[-1] + y_pred.clone().detach()
            x_history.append(
                new_x
            )

            if error_over_time is None:
                error_over_time = error_function(y_pred)
                error_over_time = error_over_time.reshape((1, -1))
            else:
                error_over_time = torch.concat([
                    error_over_time,
                    error_function(y_pred).reshape((1, -1))
                ])
        if logging:
            print(error_over_time.reshape((-1)))
        return error_function(x_history[-1] + y_pred).reshape((1, -1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        error = self.forward(x, y)
        loss = error.mean()
        return loss
