import torch.nn as nn
import torch
from torch import autograd

class Model:
    def __init__(self):
        layers = []
        layers.append(nn.Linear(2, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.lr = torch.tensor([0.0003])

    def forward(self, x_in, y):
        x = x_in
        loss_fn = nn.MSELoss()
    
        for i in list(self.net.modules())[0]:
            prev_x = x
            x =  i(x)
            if isinstance(i, nn.Linear):
                y_pred = (x.sum(dim=1) + 0.5)
                loss = loss_fn(y_pred, y)
                grad = autograd.grad(
                    outputs=loss, 
                    inputs=prev_x, 
                )[0].mean()
                i.weight = nn.Parameter(i.weight - self.lr * grad)
        return x
    
if __name__ == "__main__":
    model = Model()
    for i in range(1_000):
        out = model.forward(
            torch.tensor([
                [
                    0, 0
                ],
                [
                    0, 1
                ],
                [
                    1, 0
                ],
                [
                    1, 1
                ]
            ], dtype=torch.float, requires_grad=True).float(), 
            torch.tensor([
                0,
                1,
                1,
                1
            ]).float()
        )
        if i % 100 == 0:
            print(out)
