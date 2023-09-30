from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
import random
from helpers.get_plot_path import get_plot_path, dump_agent_data
import torch.nn as nn
import torch


class Agent:
    def __init__(self, state_size, action) -> None:
        self.model = nn.Sequential(*[
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, action),
            nn.Softmax(dim=1),
        ])
        self.sample_model = nn.Sequential(*[
            nn.Linear(state_size + 1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, state_size + 1),
            nn.Tanh()
        ])
        self.optimizer_sample_model = torch.optim.Adam(
            self.sample_model.parameters(),
            lr=1e-3
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )
        self.is_training = True
        self.accumulated_reward = 0

    def eval(self):
        self.is_training = False
        return self

    def forward(self, env: TicTacToe):
        state_actions = []
        while not env.done:
            assert env.player == 1
            state = torch.from_numpy(env.state).reshape((1, -1)).float()
            distro = self.model(state.float())
            action = None
            for i in range(distro.shape[-1]):
                if not i in env.legal_actions:
                    distro[0][i] = 0
            distro = torch.distributions.Categorical(distro)
            while action not in env.legal_actions:
                action = distro.sample()
            env.play(action)
            reward = env.winner
            state_actions.append(
                (
                    state, torch.tensor([[action]]), (
                        -1 if reward is None else reward * 10
                    ),
                    torch.from_numpy(env.state).reshape((1, -1)).float()
                )
            )
        return state_actions

    def train(self, env: TicTacToe):
        soft_reward = 0
        state_actions = self.forward(env)
        if env.winner:
            soft_reward = env.winner

        loss = 0
        loss_sample_model = 0
        if self.is_training:
            gamma = 0.998
            self.optimizer.zero_grad()
            self.optimizer_sample_model.zero_grad()
            # train sample model
            for _, (state, action, reward, next_state) in enumerate(state_actions):
                sample_next_state = self.sample_model(
                    torch.concat((action, state), dim=1))
                sample_reward = sample_next_state[0][0]
                sample_state = sample_next_state[0][1:]
                loss_sample_model += (sample_reward - reward) ** 2 + \
                    ((sample_state - next_state) ** 2).sum()
            # train q model
            for _ in range(1_00):
                (state, action, _, _) = state_actions[random.randint(
                    0, len(state_actions) - 1)]
                sample_model = self.sample_model(
                    torch.concat((action, state), dim=1))
                sample_reward = sample_model[0][0]
                sample_state = sample_model[0][1:].detach().clone()
                sample_state[sample_state < 0] = -1
                sample_state[sample_state > 0] = 1
                next_state = self.model(sample_state.reshape((1, -1)))
                # gamma * next reward
                predicted_next_reward = gamma * \
                    next_state[0][torch.argmax(next_state, dim=-1)]
                error_reward = sample_reward + predicted_next_reward - \
                    self.model(state)[0][action]
                loss += (self.model(state)[0][action] - error_reward) ** 2
            loss.backward()
            loss_sample_model.backward()
            self.optimizer.step()
            self.optimizer_sample_model.step()

        self.accumulated_reward += soft_reward
        return loss


if __name__ == "__main__":
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Agent(4*4, env.action_space)
    epochs = 10_000

    agent_y = []
    for epochs in range(epochs):
        loss = agent.train(env)
        env.reset()

        agent_y.append(agent.accumulated_reward)

        if epochs % 1_000 == 0:
            print(epochs, loss)

    random_y = []
    for epochs in range(epochs):
        while not env.done:
            env.play(random.sample(env.legal_actions, k=1)[0])

        reward = 0 if env.winner is None else env.winner
        prev = 0 if (len(random_y) == 0) else random_y[-1]
        accumulated = reward + prev

        random_y.append(
            accumulated
        )

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    plt.plot(agent_y, label="agent")
    plt.plot(random_y, label="random")
    plt.legend(loc="upper left")
    plt.savefig(get_plot_path(__file__))
    dump_agent_data(
        __file__,
        {
            "agent": agent_y,
            "random": random_y
        }
    )
