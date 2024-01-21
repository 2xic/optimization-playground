"""
tiny model for testing 
"""
import torch.nn as nn
from config import Config
import torch
import torch.optim as optim
from .model import ModelMethods

class BaseModel(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()

    # TODO: Should take in action space as 
    self.representation = nn.Sequential(*(
      nn.Linear(config.state_size, 8),
      nn.ReLU(),
      nn.Linear(8, 16),
      nn.ReLU(),
      nn.Linear(16, config.state_representation_size),
      nn.Sigmoid(),      
    ))
    self.transition_model = nn.Sequential(*[
      nn.Linear(config.state_representation_size + config.num_actions, 16),
      nn.ReLU(),
      nn.Linear(16, 16),
      nn.ReLU(),
      nn.Linear(16, config.state_representation_size),
      nn.Sigmoid(),
    ])
    self.reward_actions = nn.Sequential(*[
      nn.Linear(config.state_representation_size, 16),
      nn.ReLU(),
      nn.Linear(16, 1),
      nn.Sigmoid(),
    ])
    self.projector_network = nn.Sequential(*[
      nn.Linear(config.state_representation_size, config.projection_network_output_size),
      nn.Sigmoid(),
    ])
    # I think this has to be the same ?
    self.predictor = nn.Sequential(*[
      nn.Linear(config.projection_network_output_size, config.projection_network_output_size),
      nn.Sigmoid(),
    ])

    self.policy_predictions = nn.Sequential(*[
      nn.Linear(config.state_representation_size, config.num_actions),
      nn.Sigmoid(),
    ])

class Model(ModelMethods):
  def __init__(self, config: Config):
    self.model = BaseModel(config)
    self.config = config

  def get_policy_predictions(self, x: torch.Tensor):
    if len(x.shape) == 1:
      x = x.reshape((1, -1))
    return self.model.policy_predictions(x)

  def get_state_reward(self, x: torch.Tensor):
    return self.model.reward_actions(x)

  def get_next_state(self, state:torch.Tensor, action: int) -> torch.Tensor: 
    action_tensor = torch.zeros((1, self.config.num_actions))
    action_tensor[0][action] = 1
    state_reshaped = state.reshape((1, -1))
    combined = torch.cat(
        (state_reshaped, action_tensor),
        dim=1
    ).float()
    return self.model.transition_model(combined)

  def encode_state(self, x:torch.Tensor) -> torch.Tensor: 
    return self.model.representation(x)

  def get_optimizer(self):
    #print(list(self.model.parameters()))
    return optim.Adam(self.model.parameters(), lr=self.config.lr)    

  def get_state_projection(self, x):
      return self.model.projector_network(x)

  def get_state_prediction(self, x):
      prediction = self.model.projector_network(x)
      return self.model.predictor(prediction)

  def get_policy(self, x):
    pass
