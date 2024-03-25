"""
tiny model for testing 
"""
from typing import List, Callable
from tinygrad import Tensor, nn
from torch import Tensor as TorchTensor
from config import Config
from .model import ModelMethods
import torch
from functools import wraps
import time
from collections import defaultdict

function_time = defaultdict(float)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        function_time[func.__name__] += total_time
        
        return result
    return timeit_wrapper

class Model(ModelMethods):
  def __init__(self, config: Config):
    # Not even sure how the cuda stuff works in tinygrad yet
    # USED OT ALLOCATE TORCH TENSOR CORRECTLY
    self.device = torch.device('cpu')
    # TODO: Should take in action space as 
    self.representation: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(config.state_size, 8),
      lambda x: x.sigmoid(),
      nn.Linear(8, 16),
      lambda x: x.sigmoid(),
      nn.Linear(16, config.state_representation_size),
      lambda x: x.sigmoid(),
    ]
    self.transition_model: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(config.state_representation_size + config.num_actions, 16),
      lambda x: x.sigmoid(),
      nn.Linear(16, 16),
      lambda x: x.sigmoid(),
      nn.Linear(16, config.state_representation_size),
      lambda x: x.sigmoid(),
    ]
    self.reward_actions: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(config.state_representation_size, 8),
      lambda x: x.sigmoid(),
      nn.Linear(8, 1),
      lambda x: x.sigmoid(),
    ]
    self.projector_network: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(config.state_representation_size, 8)
    ]
    # I think this has to be the same ?
    self.predictor: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(8, 8)
    ]

    self.policy_predictions: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(config.state_representation_size, config.num_actions)
    ]

    self.config = config

  @timeit
  def get_policy_predictions(self, x: TorchTensor):
    x = self._convert_torch_tensor(x)
    return x.sequential(self.policy_predictions)

  @timeit
  def get_state_reward(self, x: TorchTensor):
    x = self._convert_torch_tensor(x)
    return x.sequential(self.reward_actions)

  @timeit
  def get_next_state(self, state:TorchTensor, action: int) -> Tensor: 
    state = self._convert_torch_tensor(state)
    action_tensor = Tensor.zeros((1, self.config.num_actions))
    action_tensor[0][action] = 1
    state_reshaped = state.reshape((1, -1))
    combined = state_reshaped.cat(
        action_tensor,
        dim=1
    ).float()
    return combined.sequential(self.transition_model)

  @timeit
  def encode_state(self, x:TorchTensor) -> Tensor: 
    x = self._convert_torch_tensor(x)
#    print(x.device)
#    print(x.sequential(self.representation))
    return x.sequential(self.representation)

  @timeit
  def get_optimizer(self):
    parameters = nn.state.get_parameters(self )
    return nn.optim.Adam(parameters, lr=self.config.lr)    

  @timeit
  def _convert_torch_tensor(self, x: TorchTensor) -> Tensor:
    if torch.is_tensor(x):
      return Tensor(x.numpy())
    return x

  @timeit
  def get_state_projection(self, x):
      return x.sequential(self.projector_network)

  @timeit
  def get_state_prediction(self, x):
      prediction = x.sequential(self.projector_network)
      return prediction.sequential(self.predictor)

  @timeit
  def get_tensor_from_array(self, x):
    if torch.is_tensor(x):
      x = x.numpy()
    return Tensor(x)

  @timeit
  def get_kl_div_loss(self, x, y):
    x = self._convert_torch_tensor(x)
    y = self._convert_torch_tensor(y)
    return (y * (y.log() - x)).mean()

  @timeit
  def get_l1_loss(self, x, y):
    x = self._convert_torch_tensor(x)
    y = self._convert_torch_tensor(y)
    return (x - y).mean()

  @timeit
  def get_l2_loss(self, x, y):
    x = self._convert_torch_tensor(x)
    y = self._convert_torch_tensor(y)
    return ((x - y) ** 2).mean()
 