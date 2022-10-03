import torch
from model import Model

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
    
class ModelComponents(torch.nn.Module):
    def __init__(self, state_size) -> None:
        super(ModelComponents, self).__init__()

        self.encoder_size = 4
        self.encoder_model = TinyModel(state_size, self.encoder_size)

        self.value_model = TinyModel(self.encoder_size, 1)
        self.transition_model = TinyModel(self.encoder_size + 1, self.encoder_size)
        self.outcome_model = TinyModel(self.encoder_size + 1, 2)

    def transition(self, state, action):
        combined = torch.concat([state, action], dim=1)
        return self.transition_model(combined)[0]
    
    def value(self, state):
        return self.value_model(state)[0]
    
    def encode(self, state):
        return self.encoder_model(state)[0]
    
    def outcome(self, state, action):
        combined = torch.concat([state, action], dim=1)
        outcome = self.outcome_model(combined)

        return outcome[0, 0], outcome[0, 1] 

class TorchModel(Model):
    def __init__(self, state_size=2) -> None:
        self.model = ModelComponents(state_size)
        self.state_size = state_size

    def core(self, state, action):
        model_state = self.encode(state) if state.shape[-1] == self.state_size else state
        reward, gamma = self.outcome(model_state, action)
        value = self.value(model_state)
        state_transition = self.transition(state, action)

        return (
            reward, gamma, value, state_transition
        )

    def transition(self, state, action):
        state = self._reshape(state)
        action = self._reshape(action)
        return self.model.transition(state, action)
    
    def value(self, state):
        state = self._reshape(state)
        return self.model.value(state)
    
    def encode(self, state):
        assert state.shape[-1] == 2
        state = self._reshape(state)
        return self.model.encode(state)
    
    def outcome(self, state, action):
        state = self._reshape(state)
        action = self._reshape(action)
        return self.model.outcome(state, action)

    def _reshape(self, x):
        if type(x) == int:
            x = torch.tensor([x])
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        return x.float()

