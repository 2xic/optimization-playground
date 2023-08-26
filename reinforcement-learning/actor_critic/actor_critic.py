"""
Simple actor critic
"""
import torch
import gymnasium as gym
from typing import List
from optimization_playground_shared.rl.actor_critic import ActorCritic, ValueCritic, Episode, loss

device = torch.device('cpu')
env = gym.make('CartPole-v1')
env.reset(seed=42)

def train():
    agent = ActorCritic(
        state_size=4,
        action_size=2
    )
    for _ in range(10_000):
        state, _ = env.reset()
        sum_reward = 0

        predictions: List[ValueCritic] = []
        episode = Episode()
        for _ in range(10_000):
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

        calculated_loss = loss(agent, predictions, episode, device=device)
        print(f"sum reward: {sum_reward} loss: {calculated_loss}")


if __name__ == "__main__":
    train()
