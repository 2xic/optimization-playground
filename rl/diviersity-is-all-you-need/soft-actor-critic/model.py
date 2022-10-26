from cmath import polar
from email import policy
from math import nextafter
from xml.sax.xmlreader import InputSource
import torch
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer


class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        assert len(x.shape) == 2, f"wrong shape, got {x.shape}"

        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.sigmoid(self.l2(x))
        x = torch.nn.functional.sigmoid(self.l3(x))
#        x = self.l3(x)

        return x


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetwork, self).__init__()
    
        self.linear1 = torch.nn.Linear(num_inputs, 128)
        self.linear2 = torch.nn.Linear(128, 128)

        self.mean_linear = torch.nn.Linear(128, num_actions)
        self.log_std_linear = torch.nn.Linear(128, num_actions)

    def forward(self, state):
        self.log_std_min = -2
        self.log_std_max = 2

        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state):
        (mean, log_std) = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std*z)
        action = action.cpu()
        return action


class SAC:
    def __init__(self, input_size, output_size):
        self.action_network = PolicyNetwork(input_size, output_size)
        self.q_1 = SimpleModel(input_size, output_size)
        self.q_2 = SimpleModel(input_size, output_size)

        self.q_1_target = SimpleModel(input_size, output_size)
        self.q_2_target = SimpleModel(input_size, output_size)
        self.update_target(tau=1)

        lr = 1e-4
        self.optimizer_q_1 = torch.optim.Adam(self.q_1.parameters(), lr=lr)
        self.optimizer_q_2 = torch.optim.Adam(self.q_2.parameters(), lr=lr)
        self.optimizer_action = torch.optim.Adam(
            self.action_network.parameters(), lr=lr)
        self.gamma = 0.7
        self.alpha = 0.4

    def optimize(self, replay: ReplayBuffer):
        # TODO: sample this from replay

        self.optimizer_action.zero_grad()
        self.optimizer_q_1.zero_grad()
        self.optimizer_q_2.zero_grad()

        total_error_policy = torch.zeros(1)
        total_error_q = torch.zeros(1)
        for _ in range(256):
            event = replay.sample()
            state, reward, next_state, is_last_state = event.state, event.reward, event.metadata[
                'next_state'], event.is_last_state
            next_state = self.reshape_tensor(next_state)
            state = self.reshape_tensor(state)

            # update q
            for model, optimizer in zip([self.q_1, self.q_2], [self.optimizer_q_1, self.optimizer_q_2]):
                model.zero_grad()
                state = state.float()
                next_state = next_state.float()

                network_value = model(state)
                target_q = None
                with torch.no_grad():
                    target_q = self.y(reward, next_state, is_last_state)

                assert not torch.isnan(target_q)
                assert not torch.any(network_value == torch.nan)
                assert not torch.any(target_q == torch.nan)

                error = ((network_value - target_q) ** 2).sum()
                error.backward()
                # optimizer.step()
                total_error_q += error

            # update policy
            error = self.q_policy_error(state, use_target=False)
            error.backward()
            total_error_policy += error

        self.optimizer_action.step()
        self.optimizer_q_1.step()
        self.optimizer_q_2.step()

        self.update_target()

        return {
            "total_error_policy": total_error_policy.item(),
            "total_error_q": total_error_q.item(),
        }

    def q_policy_error(self, next_state, use_target):
        next_state = self.reshape_tensor(next_state)
        min_q, policy = self.min_q(next_state, use_target)
        assert not torch.isnan(min_q)
        assert not torch.isnan(policy), policy
        return min_q - self.alpha * policy

    def y(self, reward, next_state, is_last_state):
        next_state = self.reshape_tensor(next_state)
        return reward + self.gamma * (
            1 - int(is_last_state)
        ) * self.q_policy_error(next_state, use_target=True)

    def min_q(self, next_state, use_target):
        next_state = self.reshape_tensor(next_state)
        action, policy = self.action(next_state)
        min_q = None
        with torch.no_grad():
            min_q = torch.min(
                self.q_1_target(next_state)[0][action] if use_target else self.q_1(
                    next_state)[0][action],
                self.q_2_target(next_state)[0][action] if use_target else self.q_2(
                    next_state)[0][action]
            )

        return min_q, (policy[0][action])

    def action(self, state):
        state = self.reshape_tensor(state)
        policy = self.action_network.get_action(state.float())
#        print(policy.shape)
#        exit(0)
        # torch.distributions.Categorical(policy).sample()
        action = torch.argmax(policy, dim=1)
        return action, policy

    def update_target(self, tau=0.4):
        for target, local in zip([self.q_1_target, self.q_2_target], [self.q_1, self.q_2]):
            for target_param, local_param in zip(target.parameters(), local.parameters()):
                target_param.data.copy_(
                    tau*local_param.data + (1.0-tau)*target_param.data)

    def reshape_tensor(self, x):
        if len(x.shape) == 1:
            return x.reshape((1, ) + x.shape)
        return x
