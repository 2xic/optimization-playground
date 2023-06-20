from torch import nn

class HyperNetwork(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.out = nn.Sequential( 
            nn.Linear(z, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 4 * 4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.out(x)
        return x.view(16, 16, 4, 4)
