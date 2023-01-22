"""
Section 6.7 -> page 136
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.Epsilon import EpsilonGreedy
from helpers.action_policy.softmax_soft_policy import SoftmaxSoftPolicy
import random
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
from helpers.State import State
import os
import numpy as np

class Double_Q_learning:
    def __init__(self, action, eps=1, decay=0.9999) -> None:
        self.q_1 = State(action)
        self.q_2 = State(action)
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=eps,
            decay=decay,
            search=self.search
        )
        self.is_training = True
        self.softmax = SoftmaxSoftPolicy()

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return self.softmax((
            self.q_1[str(self.env)].np() + 
            self.q_2[str(self.env)].np()
        ), legal_actions=self.env.legal_actions)

    def get_action(self):
        action = self.epsilon(self)
        return action

    def train(self, env: TicTacToe):
        alpha = 0.4
        gamma = 0.8
        self.env = env

        sum_rewards = 0
        while not env.done:
            state = str(env.state)
            action = self.get_action()
            reward = env.play(action)

            next_state = str(env.state)

            if 0.5 < np.random.rand():
                self.q_1[state][action] += alpha * (
                    reward + gamma * 
                    self.q_2[next_state][
                        self.q_1[next_state].argmax()
                    ] - 
                    self.q_1[state][action]
                )
            else:
                self.q_2[state][action] += alpha * (
                    reward + gamma * 
                    self.q_1[next_state][
                        self.q_2[next_state].argmax()
                    ] - 
                    self.q_2[state][action]
                )
            sum_rewards += reward
        return sum_rewards

if __name__ == "__main__":
    play_tic_tac_toe(Double_Q_learning, dirname=os.path.dirname(os.path.abspath(__file__)))
    