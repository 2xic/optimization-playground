from dataclasses import dataclass
from typing import Optional, Union, Dict
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

    @classmethod
    def image_prediction(cls, path):
        with open(path, "rb") as image:
            f = image.read()
            b = bytearray(f).hex()
            return Prediction(
                value="0x" + b,
                prediction_type="image"
            )

@dataclass
class Metrics:
    epoch: int
    loss: Union[float, Dict[str, float]]
    training_accuracy: Optional[float] = None
    testing_accuracy: Optional[float] = None
    timestamp: float=time.time()
    prediction: Prediction = None
    
    def __init__(self, epoch: int, loss: Union[float, Dict[str, float]], training_accuracy=None, testing_accuracy=None, timestamp=time.time(), prediction=None):
        self.epoch = epoch
        if type(loss) == float:
            self.loss = loss
        else:
            self.loss = {
                key:self._get_tensor_float(value)
                for key, value in loss.items()
            }
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
