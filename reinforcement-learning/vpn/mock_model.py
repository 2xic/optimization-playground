import torch
from model import Model

class MockModel(Model):
    def __init__(self) -> None:
        pass

    def core(self, _state, _action):
        reward, gamma, value, state_transition = torch.rand(
            1), torch.rand(1), torch.rand(1), torch.rand(3)

        return (
            reward, gamma, value, state_transition
        )

    def transition(self, _state, _action):
        return torch.rand(3)

    def outcome(self, _state, _action):
        return torch.rand(1), torch.rand(1)

    def value(self, _state):
        return torch.rand(1)

    def encode(self, _state):
        return torch.rand(3)
