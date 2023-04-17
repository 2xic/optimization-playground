from helpers.action_policy.Epsilon import EpsilonGreedy
from helpers.State import State
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
import os
import random
from helpers.action_policy.ArgMaxTieBreak import argmax_tie_break
import numpy as np

class n_step_sarsa:
    def __init__(self, action) -> None:
        self.alpha = 0.1
        self.gamma = 0.8

        self.epsilon = EpsilonGreedy(
            actions=action,
            eps=0.1,
            search=self.search,
        )
        self.n = 4
        self.q = State(action)

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return argmax_tie_break(
            self.q[str(self.env.state)].np().astype(np.float),
            non_max=self.env.legal_actions,
        )
        return self.softmax(self.q[str(self.env.state)].np(), legal_actions=self.env.legal_actions)

    def get_action(self):
        return self.epsilon(
            self
        )

    def train(self, env):
        T = float('inf')
        theta = None
        t = 0
        rewards = []
        states = []
        self.env = env

        action = self.get_action()
        actions = []

        while theta != T - 1:
            if t < T:
                states.append(str(env.state))
                actions.append(action)

                reward = env.play(action)
                rewards.append(reward)
                if env.done:
                    T = t + 1
                else:
                    action = self.epsilon(self)
            theta = t - self.n 
            if theta >= 0:
                G = 0
                for i in range(theta + 1, min(theta + self.n ,T)):
                    G += self.gamma ** (i - theta - 1) * rewards[i]
                if theta + self.n < T:
                    G += self.gamma ** self.n * self.q[
                        states[theta + self.n]
                    ][actions[theta + self.n]]
                self.q[states[theta]][actions[theta]] += self.alpha * (G - self.q[states[theta]][actions[theta]])
            t += 1

if __name__ == "__main__":
    play_tic_tac_toe(n_step_sarsa, dirname=os.path.dirname(os.path.abspath(__file__)))
