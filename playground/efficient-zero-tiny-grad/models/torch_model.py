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
      nn.ELU(),
      nn.Linear(8, 16),
      nn.ELU(),
      nn.Linear(16, config.state_representation_size),
      nn.ELU(),      
    ))
    self.transition_model = nn.Sequential(*[
      nn.Linear(config.state_representation_size + config.num_actions, 16),
      nn.ELU(),
      nn.Linear(16, 16),
      nn.ELU(),
      nn.Linear(16, config.state_representation_size),
      nn.Sigmoid(),
    ])
    self.reward_actions = nn.Sequential(*[
      nn.Linear(config.state_representation_size, 16),
      nn.ELU(),
      nn.Linear(16, 1),
      nn.Tanh(),
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
      nn.Linear(config.state_representation_size, 16),
      nn.ELU(),
      nn.Linear(16, 32),
      nn.ELU(),
      nn.Linear(32, config.num_actions),
      nn.Softmax(dim=1),
    ])

class Model(ModelMethods):
  def __init__(self, config: Config):
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model = BaseModel(config).to(self.device)

  def get_policy_predictions(self, x: torch.Tensor):
    x = x.to(self.device)
    if len(x.shape) == 1:
      x = x.reshape((1, -1))
    assert len(x.shape) == 2
    return self.model.policy_predictions(x)

  def get_state_reward(self, x: torch.Tensor):
    x = x.to(self.device)
    assert len(x.shape) == 2
    return self.model.reward_actions(x)

  def get_next_state(self, state:torch.Tensor, action: int) -> torch.Tensor: 
    state = state.to(self.device)
    action_tensor = torch.zeros((1, self.config.num_actions), device=self.device)
    action_tensor[0][action] = 1
    state_reshaped = state.reshape((1, -1))
    combined = torch.cat(
        (state_reshaped, action_tensor),
        dim=1,
     #   device=self.device,
    ).float()
    assert len(combined.shape) == 2
    return self.model.transition_model(combined)

  def encode_state(self, x:torch.Tensor) -> torch.Tensor: 
    x = x.to(self.device)
    assert len(x.shape) == 2
    return self.model.representation(x)

  def get_optimizer(self):
    #print(list(self.model.parameters()))
    return optim.Adam(self.model.parameters(), lr=self.config.lr)    

  def get_state_projection(self, x):
      assert len(x.shape) == 2
      x = x.to(self.device)
      return self.model.projector_network(x)

  def get_state_prediction(self, x):
      assert len(x.shape) == 2
      x = x.to(self.device)
      prediction = self.model.projector_network(x)
      return self.model.predictor(prediction)

  def get_policy(self, x):
    pass

  def get_tensor_from_array(self, x):
    return torch.tensor(x)

  def get_kl_div_loss(self, x, y):
    predicted_policy_loss = nn.KLDivLoss()(
        x,
        y,
    )
    return predicted_policy_loss

  def get_l1_loss(self, x, y):
    return nn.L1Loss()(x, y)
 
  def get_l2_loss(self, x, y):
    return nn.MSELoss()(x, y)
 