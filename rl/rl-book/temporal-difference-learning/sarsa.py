"""
Section 6.3 -> page 130
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.Epsilon import EpsilonGreedy
from helpers.action_policy.softmax_soft_policy import SoftmaxSoftPolicy
import random
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
from helpers.play_grid_world_vs_random_agent import play_grid_world
from helpers.State import State
import os
from helpers.action_policy.ArgMaxTieBreak import argmax_tie_break
import numpy as np

class Sarsa:
    def __init__(self, action, eps=1, decay=0.9999) -> None:
        self.q = State(action)
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=eps,
            decay=decay,
            search=self.search
        )

        self.is_training = True

        self.state_actions_reward_pairs = []
        self.accumulated_reward = 0
        self.softmax = SoftmaxSoftPolicy()
        self.alpha = 0.7
        self.gamma = 0.8

    def eval(self):
        self.is_training = False
        return self

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        p = self.q[str(self.env.state)].np().astype(np.float)
        for index in range(len(p)):
            if index not in self.env.legal_actions:
                p[index] = float('-inf')
        return argmax_tie_break(
            p
        )
        return self.softmax(self.q[str(self.env.state)].np(), legal_actions=self.env.legal_actions)

    def get_action(self):
        return self.epsilon(
            self
        )

    def train(self, env: TicTacToe):
        self.env = env

        action = self.get_action()
        state = str(env.state)
        sum_rewards = 0
        while not env.done:
            reward = env.play(action)

            next_state = str(env.state)
            next_action = -1

            if not env.done:
                next_action = self.get_action()

            if self.is_training:# and next_action != -1:
                self.q[state][action] += self.alpha * (
                    reward + self.gamma * self.q[next_state][next_action] -
                                self.q[state][action]
                )
            action = next_action
            state = next_state
            sum_rewards += reward
        return sum_rewards

if __name__ == "__main__":
    play_tic_tac_toe(Sarsa, dirname=os.path.dirname(os.path.abspath(__file__)))
    #play_grid_world(Sarsa, dirname=os.path.dirname(os.path.abspath(__file__)))

