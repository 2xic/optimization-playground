"""
Section 6.6 -> page 133
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


class ExpectedSarsa:
    def __init__(self, action) -> None:
        self.q = State(action)
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=1,
            decay=0.9999,
            search=self.search
        )
        self.is_training = True
        self.softmax = SoftmaxSoftPolicy()
        self.alpha = 0.8
        self.gamma = 0.8

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return self.softmax(self.q[str(self.env)].np(), legal_actions=self.env.legal_actions)

    def get_action(self):
        action = self.epsilon(self)
        return action

    def train(self, env: TicTacToe):
        self.env = env
        sum_reward = 0
        while not env.done:
            state = str(env.state)
            action = self.get_action()
            reward = env.play(action)

            next_state = str(env.state)

            self.q[state][action] += self.alpha * (
                reward + self.gamma * sum(
                    [
                        self.q[next_state][action] *
                        self.softmax.softmax(self.q[next_state].np())[action]
                        for action in range(env.action_space)
                    ]
                ) -
                self.q[state][action]
            )
            sum_reward += reward

        return sum_reward


if __name__ == "__main__":
    play_tic_tac_toe(ExpectedSarsa, dirname=os.path.dirname(
        os.path.abspath(__file__)))
