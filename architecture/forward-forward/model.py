import torch.nn as nn
import torch

class FfLinear(nn.Linear):
    def __init__(self, input, output) -> None:
        super().__init__(input, output)

        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=0.003
        )
#        self.threshold = 0.5
        self.threshold = 2

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        return torch.relu(super().forward(x))
    
    def forward_score(self, x):
        return (self.forward(x) ** 2).mean(dim=1)
    
    def train(self, positive, negative):
        p_forward = self.forward(positive)
        n_forward = self.forward(negative)
        positive_goodness = ((p_forward) ** 2).mean(dim=1)#.sum(dim=0)
        negative_goodness = ((n_forward) ** 2).mean(dim=1)#.sum(dim=0)
    
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
        score = torch.zeros((x.shape[0]), device=self.device, dtype=torch.float)
        for i in list(self.net.modules())[0]:
            if isinstance(i, FfLinear):
                score += i.forward_score(x)
                x = i.forward(x)
            else:
                x = i(x)
        return score

    def train(self, positive, negative):
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        error = None
        for i in list(self.net.modules())[0]:
            if isinstance(i, FfLinear):
                positive, negative, error = i.train(positive, negative)
            else:
                positive, negative = i(positive), i(negative)
        return error
    
    def p_positive(self, x):
        predicted = ((x ** 2).sum(dim=1)) - self.threshold
        print(predicted)
        return nn.MSELoss()(torch.tensor([1]),  predicted)

    def p_negative(self, x):
        predicted = (((x ** 2).sum(dim=1)) + self.threshold)
        print(predicted)
        return nn.MSELoss()(torch.tensor([0]), predicted)
