"""
Section 6.2 -> page 130
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

class Sarsa:
    def __init__(self, action) -> None:
        self.q = State(action)
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=1,
            decay=0.999,
            search=self.search
        )

        self.is_training = True

        self.state_actions_reward_pairs = []
        self.accumulated_reward = 0
        self.softmax = SoftmaxSoftPolicy()

    def eval(self):
        self.is_training = False
        return self

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return self.softmax(self.q[str(self.env)].np(), legal_actions=self.env.legal_actions)

    def get_action(self):
        return self.epsilon(
            self
        )

    def train(self, env: TicTacToe):
        alpha = 0.7
        gamma = 0.8
        self.env = env

        action = self.get_action(env)
        state = str(env.state)
        while not env.done:
            reward = env.play(action)

            next_state = str(env.state)
            next_action = -1

            if not env.done:
                next_action = self.get_action(env)
                #env.play(next_action)
            #elif env.is_done():
             #   for i in range(0, env.action_space):
              #      self.q[state][i] = 0

            if self.is_training:
                self.q[state][action] += alpha * (
                    reward + gamma * self.q[next_state][next_action] - 
                                self.q[state][action]
                )
            action = next_action
            state = next_state

if __name__ == "__main__":
    play_tic_tac_toe(Sarsa, dirname=os.path.dirname(os.path.abspath(__file__)))
    #play_grid_world(Sarsa, dirname=os.path.dirname(os.path.abspath(__file__)))
    
