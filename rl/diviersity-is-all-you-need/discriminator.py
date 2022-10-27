import torch

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        assert len(x.shape) == 2, f"wrong shape, got {x.shape}"

        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.sigmoid(self.l2(x))
        x = torch.nn.functional.sigmoid(self.l3(x))
#        x = self.l3(x)

        return x

