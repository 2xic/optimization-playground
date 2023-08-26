import torch

class TensorDatasetBuilder:
    def __init__(self) -> None:
        self.dataset = None

    def add(self, tensor: torch.Tensor):
        tensor = tensor.unsqueeze(0)
        if self.dataset is None:
            self.dataset = tensor
        else:
            self.dataset = torch.concat((
                self.dataset,
                tensor
            ), dim=1)
