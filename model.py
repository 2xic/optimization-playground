import torch

from attention import AttentionLayer

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.blocks = [
            AttentionLayer(1, 100),
            torch.nn.Linear(100, 100)
        ] * 3

    def forward(self, x):
        x = x
        for i in self.blocks:
            x = torch.nn.functional.normalize(i(x) + x)
        print(len(self.blocks))
        return x

shape = (Model()(torch.zeros(1, 100)).shape)
print(shape)

