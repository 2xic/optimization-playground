"""
tiny model for testing 
"""
from typing import List, Callable
from tinygrad import Tensor, nn
from config import Config

class Model:
  def __init__(self, config: Config):
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

  def get_policy_predictions(self, x: Tensor):
    return x.sequential(self.policy_predictions)

  def get_state_reward(self, x: Tensor):
    return x.sequential(self.reward_actions)

  def get_next_state(self, state:Tensor, action: int) -> Tensor: 
    action_tensor = Tensor.zeros((1, self.config.num_actions))
    action_tensor[0][action] = 1
    state_reshaped = state.reshape((1, -1))
    combined = state_reshaped.cat(
        action_tensor,
        dim=1
    ).float()
    return combined.sequential(self.transition_model)

  def encode_state(self, x:Tensor) -> Tensor: 
    return x.sequential(self.representation)

  def get_optimizer(self):
    parameters = nn.state.get_parameters(self.model)
    return nn.optim.Adam(parameters, lr=0.001)
