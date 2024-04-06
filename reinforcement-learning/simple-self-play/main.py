"""
Simple actor critic
"""
from optimization_playground_shared.plot.Plot import Plot, Figure
import torch
import gymnasium as gym
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import random
from optimization_playground_shared.rl.actor_critic import Episode, eps

class Agent(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.actor(x)

device = torch.device('cuda:0')
env = gym.make('CartPole-v1')
env.reset(seed=42)

EPOCHS = 1_000
sum_append = lambda x, y: (x[-1] if len(x) > 0 else 0) + y 

def random_agents():
    scores = []
    for _ in range(EPOCHS):
        _, _ = env.reset()
        sum_reward = 0
        for _ in range(EPOCHS):
            action = random.randint(0, 1)
            _, reward, done, _, _ = env.step(action)
            sum_reward += reward
            if done:
                break
        scores.append(sum_append(scores, sum_reward))
    return scores

def train():
    agent = Agent(
        state_size=4,
        action_size=2
    ).to(device)
    scores = []
    optimizer = torch.optim.Adam(agent.parameters())
    for epochs in range(EPOCHS):
        state, _ = env.reset()
        sum_reward = 0

        loss = 0
        episode = Episode()
        action_estimate = []
        for _ in range(EPOCHS):
            actor = agent.forward(
                torch.from_numpy(state).unsqueeze(0).to(device)
            )
            dist = Categorical(actor)
            action = dist.sample()
            state, reward, done, _, _ = env.step(action.item())
            sum_reward += reward
            episode.add(reward)
            action_estimate.append(actor[0][action])
            if done:
                break
        returns = episode.get_value_reward()
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for index, i in enumerate(returns):
            # loss is how much this this reward contribute to the error 
            # reward -> 1 then we want action high
            # reward -> 0 then we want action low
            loss += (1 / action_estimate[index] * i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{epochs}: sum reward: {sum_reward} loss: {loss.item()}")
        scores.append(sum_append(scores, sum_reward))
    
    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Agent": scores,
                    "Random": random_agents(),
                },
                title="Rewards",
                x_axes_text="Epochs",
                y_axes_text="Rewards",
            )
        ],
        name='rewards.png'
    )


if __name__ == "__main__":
    train()
