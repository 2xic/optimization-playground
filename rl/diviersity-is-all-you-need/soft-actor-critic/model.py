from math import nextafter
import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.sigmoid(self.l3(x))
        return x



class SAC:
    def __init__(self, input_size, output_size):
        self.action_network = SimpleModel( input_size, output_size) 
        self.q_1 = SimpleModel( input_size, output_size)
        self.q_2 = SimpleModel( input_size, output_size)

        self.q_1_target = SimpleModel( input_size, output_size)
        self.q_2_target = SimpleModel( input_size, output_size)
        self.update()
        self.optimizer = torch.optim.Adam()
        self.gamma = 1
        self.alpha = 0.1

    def optimize(self, replay):
        # TODO: sample this from replay
        state, reward, next_state, is_last_state = None , None, None, None

        # update q
        for i in [self.q_1, self.q2]:
            network_value = i(state) 
            target_q = self.y(reward, next_state, is_last_state)
            error = (network_value - target_q) ** 2
            # TODO: backward
        # update policy
    
        pass

    def y(self, reward, next_state, is_last_state):
        min_q, policy = self.min_q(next_state)
        return reward + self.gamma * (
            1 - int(is_last_state)
        ) * min_q \
        - self.alpha * policy
    
    def min_q(self, next_state):
        next_action = self.q_1(next_state)
        action = torch.argmax(next_action)
        return torch.min(
            self.q_1_target(next_state)[action],
            self.q_1_target(next_state)[action]
        ), next_action[action]

    def action(self, state):
        return self.q_1(state)

    def update(self):
        self.q_1_target.load_state_dict(self.q_1.state_dict())
        self.q_2_target.load_state_dict(self.q_2.state_dict())

    def parameters(self):
        return self.q_1.parameters()
