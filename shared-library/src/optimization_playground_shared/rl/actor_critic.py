"""
Simple actor critic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

eps = 1e-6

class ValueCritic:
    def __init__(self, agent_value, critic_value):
        self.agent_value = agent_value
        self.critic_value = critic_value

        dist = Categorical(self.agent_value)
        self.action = dist.sample()
        self.action_prob = dist.log_prob(self.action)

    def get_action(self):
        return self.action

class Episode:
    def __init__(self):
        self.rewards = []

    def add(self, reward):
        self.rewards.append(reward)

    def get_value_reward(self):
        returns = []
        gamma = .99
        reward = 0
        for r in self.rewards[::-1]:
            reward = r + gamma * reward
            returns.insert(0, reward)
        return returns
    
class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.actor = nn.Linear(state_size, action_size)

    def forward(self, x):
        return F.softmax(self.actor(x), dim=1)


class Critic(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.critic = nn.Linear(state_size, 1)

    def forward(self, x):
        return self.critic(x)


class ActorCritic():
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )

    def forward(self, x):
        return (
            self.actor(x),
            self.critic(x)
        )


def loss(agent: ActorCritic, predictions, episode: Episode, device):
    returns = episode.get_value_reward()
    returns = torch.tensor(returns, device=device)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_losses = torch.zeros(1, device=device)
    critic_losses = torch.zeros(1, device=device)
    for value_prediction, value_R in zip(predictions, returns):
        advantage = value_R - value_prediction.critic_value.item()
        policy_losses += (-value_prediction.action_prob * advantage)
        critic_losses += (F.smooth_l1_loss(value_prediction.critic_value, torch.tensor([value_R], device=device)))

    agent.optimizer.zero_grad()
    loss = policy_losses + critic_losses
    loss.backward()

    assert agent.critic.critic.weight.grad is not None
    assert agent.actor.actor.weight.grad is not None
        
    agent.optimizer.step()
    return loss.item()
