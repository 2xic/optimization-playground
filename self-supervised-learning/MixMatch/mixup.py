import torch
from hyperparameters import alpha

class MixUp:
    def __init__(self) -> None:
        self.alpha = alpha

    """
    TODO: make this more tensor friendly
    """
    def __call__(self, x, y, x2, y2, device):
        batch_size = 1
        l = torch.distributions.beta.Beta(self.alpha, self.alpha).sample(torch.tensor([batch_size])).to(device)
        l = torch.maximum(l, torch.ones(l.shape).to(device) - l).to(device)

        new_x = x * l + (1 - l) * x2
        new_y = y * l + (1 - l) * y2

        return (
            new_x.float(),
            new_y.float()
        )
