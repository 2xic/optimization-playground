
import torch

class Prediction:
    def __init__(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
     #   assert predicted.shape == (1, ), predicted
    #    assert actual.shape == (1,), actual

        self.predicted = predicted
        self.actual = actual

    def delta(self):
        return (self.predicted - self.actual) ** 2

