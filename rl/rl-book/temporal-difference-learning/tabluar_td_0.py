"""
Section 6.1 -> page 120
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.Epsilon import EpsilonGreedy
import random
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
import os

class Tabluar_td_0:
    def __init__(self, action) -> None:
        self.v_s = State(action, value_constructor=lambda x: float(random.randint(1, 3)))
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=1,
            decay=0.999,
            search=self.search
        )

        self.is_training = True
        self.state_actions_reward_pairs = []

    def eval(self):
        self.is_training = False
        return self

    def query_state_reward(self, env):
        return [
            (i, self.v_s[str(env.soft_apply(i))])
            for i in env.legal_actions
        ]

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return list(max(self.query_state_reward(self.env), key=lambda x: x[1]))[0]

    def train(self, env: TicTacToe):
        alpha = 0.4
        gamma = 0.8
        self.env = env
        while not env.done:
            state = str(env.state)

            action = -1
            while action not in env.legal_actions:
                action = self.epsilon(
                    self
                )

            env.play(action)
            next_state = str(env.state)

            soft_reward = env.winner if env.winner is not None else 0
            reward = soft_reward * 10 if env.winner is not None else -1

            if self.is_training:
                self.v_s[state] += alpha * (
                    reward + (
                        gamma * self.v_s[next_state]
                    )
                    - self.v_s[state]
                )

if __name__ == "__main__":
    play_tic_tac_toe(Tabluar_td_0, dirname=os.path.dirname(os.path.abspath(__file__)))
