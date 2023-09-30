from collections import deque
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
        self.optimizer =  torch.optim.Adam(
            self.model.parameters(),
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
            state = torch.from_numpy(env.state).reshape((1, -1))
            distro = self.model(state.float()).log()
            action = None
            distro = torch.distributions.Categorical(distro)
            while action not in env.legal_actions:
                action = distro.sample()
            env.play(action)
            reward = env.winner
            state_actions.append(
                (
                    state, action, (
                        -1 if reward is None else reward * 10
                    ),
                    distro.log_prob(action)
                )
            )  
        return state_actions

    def train(self, env: TicTacToe):
        soft_reward = 0
        state_actions = self.forward(env)
        if env.winner:
            soft_reward = env.winner

        loss = 0
        if self.is_training:
            gamma = 0.998
            alpha = 0.001
            self.optimizer.zero_grad()
            returns = deque()
            R = 0
            for index, (_, _, reward, _) in enumerate(state_actions[::-1]):
                R = reward + gamma * R
                returns.appendleft(R)
            for index, (_, _, reward, log_prob) in enumerate(state_actions):
                future_reward_sum = 0
                for iii in range(index, len(state_actions)):
                    future_reward_sum += gamma ** (iii - index - 1) * state_actions[iii][2]
                loss += -log_prob * future_reward_sum
            loss.backward()
            self.optimizer.step()

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
        prev = 0 if(len(random_y) == 0) else random_y[-1]
        accumulated = reward + prev

        random_y.append(
            accumulated
        )

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    plt.title("Reinforce vs random agent")
    plt.plot(agent_y, label="agent")
    plt.plot(random_y, label="random")
    plt.legend(loc="upper left")
    plt.savefig(get_plot_path(__file__))
    dump_agent_data(
        __file__,
        {
            "agent":agent_y,
            "random": random_y
        }
    )
