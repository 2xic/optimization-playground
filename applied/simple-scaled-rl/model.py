import torch
import torch.nn as nn
import torch.distributions as distributions
from Epsilon import EpsilonGreedy

class SimpleQLearning(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=1)
        )
        eps=1
        decay=0.9999
        self.epsilon = EpsilonGreedy(
            eps=eps,
            decay=decay,
            actions=action_size,
        )
    
    def forward(self, state):
        state = state.reshape((1, -1))
        network_output = self.network(state)
        return network_output

    def _get_action(self, state):
        network_output = self.forward(state)
        max = distributions.Categorical(network_output[0])
        action = max.sample()
        return action
    
    def get_action(self, state):
        epsilon_action = self.epsilon()
        if epsilon_action is None:
            return self._get_action(state)
        else:
            return epsilon_action
    
    def train(self):
        pass

if __name__ == "__main__":
    action_size = 4
    state_size = 4
    state = torch.zeros((state_size))
    model = SimpleQLearning(state_size, action_size)
    print(model.get_action(state))
