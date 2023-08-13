from dataclasses import dataclass
from typing import Optional
import time
import torch

@dataclass
class Prediction:
    value: str = None
    prediction_type: str = None
    
    def __init__(self, value, prediction_type):
        self.value = value
        self.prediction_type = prediction_type

    @classmethod
    def text_prediction(cls, value):
        return Prediction(
            value=value,
            prediction_type="text"
        )

@dataclass
class Metrics:
    epoch: int
    loss: float
    training_accuracy: Optional[float] = None
    testing_accuracy: Optional[float] = None
    timestamp: float=time.time()
    prediction: Prediction = None
    
    def __init__(self, epoch: int, loss: float, training_accuracy=None, testing_accuracy=None, timestamp=time.time(), prediction=None):
        self.epoch = epoch
        self.loss = self._get_tensor_float(loss)
        self.training_accuracy = self._get_tensor_float(training_accuracy)
        self.testing_accuracy = self._get_tensor_float(testing_accuracy)
        self.timestamp = timestamp
        self.prediction = prediction

    def _get_tensor_float(self, item):
        if item is None:
            return None
        elif torch.is_tensor(item):
            return item.item()
        return item
