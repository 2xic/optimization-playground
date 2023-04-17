"""
Section 6.5 -> page 131
"""
from optimization_utils.envs.TicTacToe import TicTacToe
from helpers.State import State
from helpers.StateValue import StateValue
from helpers.action_policy.Epsilon import EpsilonGreedy
from helpers.action_policy.softmax_soft_policy import SoftmaxSoftPolicy
import random
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
from helpers.play_grid_world_vs_random_agent import play_grid_world
from helpers.State import State
import os
from helpers.action_policy.ArgMaxTieBreak import argmax_tie_break
import numpy as np

class Q_learning:
    def __init__(self, action, eps=1, decay=0.9999) -> None:
        self.q = State(action)#, value_constructor=lambda n: StateValue(n, initial_value=lambda: 0))
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=eps,
            decay=decay,
            search=self.search
        )
        self.is_training = True
        self.softmax = SoftmaxSoftPolicy()
        self.alpha = 0.8
        self.gamma = 0.8

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return argmax_tie_break(
            self.q[str(self.env.state)].np().astype(np.float),
            non_max=self.env.legal_actions,
        )

        #return (self.q[str(self.env.state)].np()).argmax()
        return self.softmax(
            self.q[str(self.env.state)].np(), 
            legal_actions=self.env.legal_actions
        )

    def get_action(self):
        action = self.epsilon(self)
        return action

    def train(self, env: TicTacToe):
        self.env = env

        sum_rewards = 0
        while not env.done:
            state = str(env.state)
            action = self.get_action()
            reward = env.play(action)

            next_state = str(env.state)

            self.q[state][action] += self.alpha * (
                reward + (self.gamma * self.q[next_state].max() -
                            self.q[state][action])
            )
            sum_rewards += reward
        return sum_rewards

if __name__ == "__main__":
    play_tic_tac_toe(Q_learning, dirname=os.path.dirname(os.path.abspath(__file__)))

