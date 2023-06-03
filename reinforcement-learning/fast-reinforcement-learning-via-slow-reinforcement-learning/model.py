import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, state_size) -> None:
        super().__init__()

        self.state_size = 9
        self.action_space = 9

        self.input_size = state_size + 3
        self.hidden_size = 20
        
        self.num_layers = 2
        self.rnn = nn.GRU(
            self.input_size, 
            self.hidden_size, 
            self.num_layers,
            batch_first = True
        ) 

        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, state_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
        x = x.float()
        assert len(x.shape) == 2
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        x, new_hidden = self.rnn(x, hidden)
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x, new_hidden

    def init_hidden(self, batch_size):
        return torch.randn(self.num_layers, batch_size, self.hidden_size).float()

if __name__ == "__main__":
    state_vector = input = torch.randn(5, 10)
    model = Model(
        10
    )
    output, hidden = model(state_vector)
    
    print(output.shape)
    print(hidden.shape)
