"""
tiny model for testing 
"""
import torch.nn as nn
from config import Config
import torch
import torch.optim as oprim

class Model:
  def __init__(self, config: Config):
    # TODO: Should take in action space as 
    self.representation = nn.Sequential(*(
      nn.Linear(config.state_size, 8),
      nn.Sigmoid(),
      nn.Linear(8, 16),
      nn.Sigmoid(),
      nn.Linear(16, config.state_representation_size),
      nn.Sigmoid(),      
    ))
    self.transition_model = nn.Sequential(*[
      nn.Linear(config.state_representation_size + config.num_actions, 16),
      nn.Sigmoid(),
      nn.Linear(16, 16),
      nn.Sigmoid(),
      nn.Linear(16, config.state_representation_size),
      nn.Sigmoid(),
    ])
    self.reward_actions = nn.Sequential(*[
      nn.Linear(config.state_representation_size, 8),
      nn.Sigmoid(),
      nn.Linear(8, 1),
      nn.Sigmoid(),
    ])
    self.projector_network = nn.Sequential(*[
      nn.Linear(config.state_representation_size, 8)      
    ])
    # I think this has to be the same ?
    self.predictor = nn.Sequential(*[
      nn.Linear(8, 8)
    ])

    self.policy_predictions = nn.Sequential([
      nn.Linear(config.state_representation_size, config.num_actions)
    ])

    self.config = config

  def get_policy_predictions(self, x: torch.Tensor):
    return self.policy_predictions(x)

  def get_state_reward(self, x: torch.Tensor):
    return self.reward_actions(x)

  def get_next_state(self, state:torch.Tensor, action: int) -> torch.Tensor: 
    action_tensor = torch.zeros((1, self.config.num_actions))
    action_tensor[0][action] = 1
    state_reshaped = state.reshape((1, -1))
    combined = state_reshaped.cat(
        action_tensor,
        dim=1
    ).float()
    return combined.sequential(self.transition_model)

  def encode_state(self, x:torch.Tensor) -> torch.Tensor: 
    return self.representation(x)

  def get_optimizer(self):
    return oprim.Adam(self.parameters())    
