"""
Section 5.4 -> page 101
"""
from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
from helpers.State import State
from helpers.action_policy.softmax_soft_policy import SoftmaxSoftPolicy
from helpers.StateValue import StateValue
import random
from helpers.get_plot_path import get_plot_path
from collections import defaultdict

class Agent:
    def __init__(self, action) -> None:
        self.p_s = State(action)
        self.q_s = State(action)
        self.n_s = State(action, value_constructor=lambda n: StateValue(n, initial_value=lambda: 0))
        self.softmax = SoftmaxSoftPolicy()

        self.is_training = True

        self.state_actions_reward_pairs = []
        self.accumulated_reward = 0
        self.actions = action
        self.eps = 0.01

    def eval(self):
        self.is_training = False
        return self
    
    def forward(self, env):
        state_actions = []
        index = 0
        while not env.done:
            assert env.player == 1                
            state = str(env.state)
            action = self.softmax(self.p_s[state].np(), env.legal_actions)
            env.play(action)
            reward = env.winner
            state_actions.append(
                (
                    state, action, (
                        -1 if reward is None else reward * 10
                    )
                )
            )  
            index += 1
        return state_actions

    def train(self, env: TicTacToe):
        soft_reward = 0
        state_actions = self.forward(env)
        if env.winner:
            soft_reward = env.winner

        if self.is_training:
            G = 0
            gamma = .99
            first_index_seen_state = defaultdict(int)
            for index, (state, _, _) in enumerate(state_actions):
                if not state in first_index_seen_state:
                  first_index_seen_state[state] = index

            for index in range(len(state_actions) - 2, -1, -1):
                (_, _, next_reward) = state_actions[index + 1]
                (state, action, _) = state_actions[index]
                # for tic tac toe this should always be hit ...
                G = gamma * G + next_reward
                if index <= first_index_seen_state[state]:
                    # simulate the avg
                    self.q_s[state][action] = max(0, self.q_s[state][action] + (G - self.q_s[state][action]) * (1 / (1 + self.n_s[state][action])))
                    self.n_s[state][action] += 1
                    # update the policy
                    state_argmax = self.q_s[state].argmax()
                    for i in range(self.actions):
                        if i == state_argmax:
                            self.p_s[state][i] = (1 - self.eps) + self.eps/self.actions
                        else:
                            self.p_s[state][i] = self.eps / self.actions
        self.accumulated_reward += soft_reward

if __name__ == "__main__":
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Agent(env.action_space)
    epochs = 10_000

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
    plt.savefig(get_plot_path(__file__))
