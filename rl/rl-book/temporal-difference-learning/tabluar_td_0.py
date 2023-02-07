"""
Section 6.1 -> page 120
"""
from optimization_utils.envs.TicTacToe import TicTacToe
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.Epsilon import EpsilonGreedy
import random
import os

class Tabluar_td_0:
    def __init__(self, action) -> None:
        self.v_s = State(action, value_constructor=lambda _: float(random.randint(1, 3)))
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=1,
            decay=0.9999,
            search=self.search
        )
        self.is_training = True
        self.state_actions_reward_pairs = []
        self.alpha = 0.4
        self.gamma = 0.8

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
        self.env = env
        sum_reward = 0
        while not env.done:
            state = str(env.state)

            action = -1
            while action not in env.legal_actions:
                action = self.epsilon(
                    self
                )

            env_reward = env.play(action)
            next_state = str(env.state)

            soft_reward = env.winner if env.winner is not None else 0
            reward = soft_reward * 10 if env.winner is not None else -1

            if self.is_training:
                self.v_s[state] += self.alpha * (
                    reward + (
                        self.gamma * self.v_s[next_state]
                    )
                    - self.v_s[state]
                )
            sum_reward += env_reward

        return sum_reward

if __name__ == "__main__":
    play_tic_tac_toe(Tabluar_td_0, dirname=os.path.dirname(os.path.abspath(__file__)))
