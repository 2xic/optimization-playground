"""
Simple actor critic
"""
import torch
import gymnasium as gym
from typing import List
from optimization_playground_shared.rl.actor_critic import ActorCritic, ValueCritic, Episode, loss
from optimization_playground_shared.plot.Plot import Plot, Figure
import random

device = torch.device('cpu')
env = gym.make('CartPole-v1')
env.reset(seed=42)


EPOCHS = 3_000

def random_agents():
    scores = []
    for epochs in range(EPOCHS):
        _, _ = env.reset()
        sum_reward = 0
        for _ in range(EPOCHS):
            action = random.randint(0, 1)
            _, reward, done, _, _ = env.step(action)
            sum_reward += reward
            if done:
                break
        if epochs % 100 == 0:
            scores.append(sum_reward)
            print(epochs)
    return scores

def train():
    agent = ActorCritic(
        state_size=4,
        action_size=2
    )
    scores = []
    for epochs in range(EPOCHS):
        state, _ = env.reset()
        sum_reward = 0

        predictions: List[ValueCritic] = []
        episode = Episode()
        for _ in range(EPOCHS):
            (actor, critic) = agent.forward(
                torch.from_numpy(state).unsqueeze(0)
            )
            predictions.append(ValueCritic(
                actor,
                critic
            ))
            action = predictions[-1].get_action()
            state, reward, done, _, _ = env.step(action.item())
            sum_reward += reward
            episode.add(reward)
            if done:
                break
        if epochs % 100 == 0:
            scores.append(sum_reward)
        calculated_loss = loss(agent, predictions, episode, device=device)
        print(f"{epochs} sum reward: {sum_reward} loss: {calculated_loss}")

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
