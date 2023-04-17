from optimization_utils.tree.State import State
import random
import torch

""" 
Same component as used in VPN
"""
class TinyModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TinyModel, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 200)
        self.l2 = torch.nn.Linear(200, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, state_size, action_space) -> None:
        super(Model, self).__init__()

        self.state_representation_size = 4
        self.action_space = action_space

        self.representation = TinyModel(state_size, self.state_representation_size)
        self.dynamics = TinyModel(self.state_representation_size + 1, self.state_representation_size + 1)
        self.prediction = TinyModel(self.state_representation_size, action_space + 1)

    def get_action(self, state: State):
        return random.sample(state.possible_actions)

    def get_state_representation(self, state: torch.Tensor):
        return self.representation(state)

    def get_dynamics(self, state: torch.Tensor, action: int):
        combined = torch.concat([state, action], dim=1)
        results = self.dynamics(combined)
        return results[:, 0], results[:, 1:]

    def get_prediction(self, state: torch.Tensor):
        policy_reward = self.prediction(state)

        return policy_reward[:, :self.action_space], policy_reward[:, -1]

