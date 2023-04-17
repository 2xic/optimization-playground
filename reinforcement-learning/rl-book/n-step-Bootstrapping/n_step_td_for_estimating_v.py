from helpers.action_policy.Epsilon import EpsilonGreedy
from helpers.State import State
from helpers.play_tic_tac_toe_vs_random_agent import play_tic_tac_toe
import os
import random

class n_step_td:
    def __init__(self, action) -> None:
        self.alpha = 0.1
        self.gamma = 0.8

        self.epsilon = EpsilonGreedy(
            actions=action,
            eps=0.1,
            search=self.search,
        )
        self.n = 4
        self.v_s = State(action, value_constructor=lambda _: 0)

    def query_state_reward(self, env):
        return [
            (i, self.v_s[str(env.soft_apply(i))])
            for i in env.legal_actions
        ]

    def search(self):
        return random.sample(self.env.legal_actions, k=1)[0]

    def on_policy(self):
        return list(max(self.query_state_reward(self.env), key=lambda x: x[1]))[0]

    def train(self, env):
        T = float('inf')
        theta = None
        t = 0
        rewards = []
        states = []
        self.env = env

        while theta != T - 1:
            if t < T:
                states.append(str(env.state))
                action = self.epsilon(self)
                reward = env.play(action)
                rewards.append(reward)
                if env.done:
                    T = t + 1
            theta = t - self.n 
            if theta >= 0:
                G = 0
                for i in range(theta + 1, min(theta + self.n ,T)):
                    G += self.gamma ** (i - theta - 1) * rewards[i]
                if theta + self.n < T:
                    G += self.gamma ** self.n * self.v_s[states[theta + self.n]]
                self.v_s[states[theta]] += self.alpha * (G - self.v_s[states[theta]])
            t += 1

if __name__ == "__main__":
    play_tic_tac_toe(n_step_td, dirname=os.path.dirname(os.path.abspath(__file__)))
