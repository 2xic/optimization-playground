"""
Section 5.7 -> page 111
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.softmax_soft_policy import SoftmaxSoftPolicy
import random


class Agent:
    def __init__(self, action) -> None:
        self.q_s = State(action)
        self.c_s = State(action)
        self.softmax = SoftmaxSoftPolicy()

        self.is_training = True

        self.state_actions_reward_pairs = []
        self.accumulated_reward = 0

    def eval(self):
        self.is_training = False
        return self

    def train(self, env: TicTacToe):
        state_actions = []
        soft_reward = 0
        while not env.done:
            assert env.player == 1
            state = str(env.state)
            action = -1
            while action not in env.legal_actions:
                action = self.softmax(self.q_s[state].np(), env.legal_actions)

            env.play(action)
            reward = env.winner
            state_actions.append(
                (
                    state, action, (
                        -1 if reward is None else reward * 10
                    )
                )
            )

        if env.winner:
            soft_reward = env.winner

        if self.is_training:
            G = 0
            W = 1
            gamma = .99
            for index in range(len(state_actions) - 2, -1, -1):
                (_, _, next_reward) = state_actions[index + 1]
                (state, action, reward) = state_actions[index]

                G = gamma * G + next_reward
                # C[S_t, A_t] += W
                self.c_s[state][action] += W
                # Q(S_t, A_t) += (W/C(S_t, A_t) * (G - Q[S_t, A_t]))
                q = self.q_s[state][action]
                c = self.c_s[state][action]
                self.q_s[state][action] += (W/c) * (G - q)

                # A_t != policy(S)
                if self.q_s[state].argmax() != action:
                    break

                # W = W * (1 / B[A_t, S_t])
                b_s_a = self.softmax.softmax(self.q_s[state].np())[action]
                W = W * 1 / b_s_a
          #  print(G)
        self.accumulated_reward += soft_reward


if __name__ == "__main__":
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Agent(env.action_space)
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
