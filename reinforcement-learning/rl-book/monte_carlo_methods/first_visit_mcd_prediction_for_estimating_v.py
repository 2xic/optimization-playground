"""
Section 5.1 -> page 114
"""
from .off_policy_mc_control import Agent
from optimization_utils.envs.TicTacToe import TicTacToe
from helpers.State import State
import numpy as np
from collections import defaultdict

def train_model():
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Agent(env.action_space)
    epochs = 5_000

    agent_y = []
    for epochs in range(epochs):
        agent.train(env)
        env.reset()

        agent_y.append(agent.accumulated_reward)

        if epochs % 1_000 == 0:
            print(epochs)
    return agent

def evaluate_policy(agent):
    value_s = defaultdict(int)
    returns_s = defaultdict(list)
    env = TicTacToe(n=4, is_auto_mode=True)
    for epochs in range(10):
        env.reset()
        state_actions = agent.forward(env)
        first_seen_state_index = {}

        for index, (state, _, _) in enumerate(state_actions):
            if state not in first_seen_state_index:
                first_seen_state_index[state] = index

        G = 0
        gamma = .99
        for index in range(len(state_actions) - 1, -1, -1):
            (state, _, next_reward) = state_actions[index]
            G = gamma * G + next_reward
            if index <= first_seen_state_index[state]:
                returns_s[state].append(G)
                value_s[state] = sum(returns_s[state])/len(returns_s[state])
    for (key, value) in sorted(value_s.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(key)
        print(value)
        print()

if __name__ == "__main__":
    evaluate_policy(train_model())

