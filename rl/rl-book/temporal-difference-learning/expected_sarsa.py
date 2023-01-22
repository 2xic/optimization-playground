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

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return self.softmax(self.q[str(self.env)].np(), legal_actions=self.env.legal_actions)

    def get_action(self):
        action = self.epsilon(self)
        return action

    def train(self, env: TicTacToe):
        alpha = 0.8
        gamma = 0.8
        self.env = env

        while not env.done:
            state = str(env.state)
            action = self.get_action()
            reward = env.play(action)

            next_state = str(env.state)

            self.q[state][action] += alpha * (
                reward + gamma * sum(
                    [
                        self.q[next_state][action] *
                        self.softmax.softmax(self.q[next_state].np())[action]
                        for action in range(env.action_space)
                    ]
                ) -
                self.q[state][action]
            )


if __name__ == "__main__":
    play_tic_tac_toe(ExpectedSarsa, dirname=os.path.dirname(
        os.path.abspath(__file__)))
