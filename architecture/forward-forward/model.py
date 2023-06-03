import torch.nn as nn
import torch


class FfLinear(nn.Linear):
    def __init__(self, input, output, p=0,lr=0.03) -> None:
        super().__init__(input, output)
        self.p = p

        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )
#        self.threshold = 0.5
        self.threshold = 2
        self.activation = torch.nn.Hardtanh(0, 5)

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        output = self.activation(super().forward(x))
        output = torch.nn.functional.dropout(output, p=self.p)
        return output

    def forward_score(self, x):
        raw = self.forward(x)
        return (raw ** 2).mean(dim=1), raw

    def train(self, positive, negative):
        positive_goodness, p_forward = self.forward_score(positive)
        negative_goodness, n_forward = self.forward_score(negative)

        """
        error = torch.mean(
            torch.cat([
                (-positive_goodness - self.threshold),
                (negative_goodness + self.threshold),
            ])
        ) 
        """
        error = torch.log(1 + torch.exp(torch.cat([
            -positive_goodness + self.threshold,
            negative_goodness - self.threshold]))
        ).mean()

        self.opt.zero_grad()

        error.backward()
        self.opt.step()

        return p_forward.detach(), n_forward.detach(), error


class Model:
    def __init__(self, layers):
        self.net = nn.Sequential(*layers)
        self.device = torch.device('cpu')

    def to(self, device):
        self.net = self.net.to(device)
        self.device = device
        return self

    def forward(self, x):
        x = x.to(self.device)
        score = torch.zeros(
            (x.shape[0]), 
            device=self.device, 
            dtype=torch.float
        )
        for i in list(self.net.modules())[0]:
            i: FfLinear = i
            goodness, x = i.forward_score(x)
            score += goodness
        return score

    def train(self, positive, negative):
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        error = None
        for i in list(self.net.modules())[0]:
            positive, negative, error = i.train(positive, negative)
        return error
