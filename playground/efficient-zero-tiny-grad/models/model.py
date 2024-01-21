from abc import ABC
from torch import Tensor


class ModelMethods(ABC):
    def get_policy_predictions(self, x:  Tensor):
        pass

    def get_state_reward(self, x:  Tensor):
        pass

    def get_next_state(self, state: Tensor, action: int) -> Tensor: 
        pass

    def encode_state(self, x: Tensor) -> Tensor: 
        pass

    def get_optimizer(self):
        pass

    def get_state_projection(self, x):
        pass

    def get_state_prediction(self, x):
        pass

    def _convert_torch_tensor(self, x:  Tensor) -> Tensor:
        pass

    