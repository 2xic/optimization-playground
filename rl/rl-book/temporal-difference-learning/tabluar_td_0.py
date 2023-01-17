"""
Section 6.1 -> page 120
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.Epsilon import EpsilonGreedy
import random

class Tabluar_td_0:
    def __init__(self, action) -> None:
        self.v_s = State(action, value_constructor=lambda x: float(random.randint(1, 3)))
        self.epsilon = EpsilonGreedy(
            actions=-1,
            eps=1,
            decay=0.01,
            search=self.search
        )

        self.is_training = True

        self.state_actions_reward_pairs = []
        self.accumulated_reward = 0

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
        agent.accumulated_reward += soft_reward


if __name__ == "__main__":
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Tabluar_td_0(env.action_space)
    epochs = 5_000

    agent_y = []
    for epochs in range(epochs):
        agent.train(env)
        env.reset()

        agent_y.append(agent.accumulated_reward)

        if epochs % 1_000 == 0:
            print(epochs)

    random_y = []
    for epochs in range(epochs):
        while not env.done:
            env.play(random.sample(env.legal_actions, k=1)[0])

        reward = 0 if env.winner is None else env.winner
        prev = 0 if(len(random_y) == 0) else random_y[-1]
        accumulated = reward + prev

        random_y.append(
            accumulated
        )

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    plt.plot(agent_y, label="agent")
    plt.plot(random_y, label="random")
    plt.legend(loc="upper left")
    plt.show()
