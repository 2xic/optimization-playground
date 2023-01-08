"""
Section 5.7 -> page 111
"""
from softmax_soft_policy import SoftmaxSoftPolicy
from optimization_utils.envs.TicTacToe import TicTacToe
import numpy as np
import matplotlib.pyplot as plt

class StateValue:
    def __init__(self, n) -> None:
        self.value = [0, ] * n

    def __setitem__(self, key, value):
        self.value[key] = value

    def __getitem__(self, key):
        return self.value[key]
    
    def np(self):
        return np.asarray(self.value)

    def argmax(self):
        return np.argmax(self.np())

class State:
    def __init__(self, n) -> None:
        self.state = {}
        self.n = n

    def __getitem__(self, key):
        if not key in self.state:
            self.state[key] = StateValue(self.n)
        return self.state[key]

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
        while not env.done:
            state = str(env.state)
            action = -1
            while action not in env.legal_actions:
                action = self.softmax(self.q_s[state].np(), env.legal_actions)        

            env.play(action)
            reward = env.winner
            state_actions.append(
                (
                    state, action, (
                        0 if reward is None else reward * 10
                    )
                )
            )    

        if self.is_training:        
            G = 0
            W = 1
            gamma = .9
            for index in range(len(state_actions) - 2, -1, -1):
                (_, _ , next_reward) = state_actions[index + 1]
                (state, action, reward)  = state_actions[index]

                G = gamma * G + next_reward
                self.c_s[state][action] += W

                q = self.q_s[state][action]
                c = self.c_s[state][action]

                self.q_s[state][action] += (W/c) * (G - q)

                if self.q_s[state].argmax() != action:
                    break
                
                b_s_a = self.softmax.softmax(self.q_s[state].np())[action]
                W = W * 1 / b_s_a
          #  print(G)
        self.accumulated_reward += state_actions[-1][-1]

if __name__ == "__main__":
    env = TicTacToe(n=3)
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
    agent = Agent(env.action_space).eval()
    for epochs in range(epochs):
        agent.train(env)
        env.reset()

        random_y.append(agent.accumulated_reward)

        if epochs % 1_000 == 0:
            print(epochs)

    plt.plot(agent_y, label="agent")
    plt.plot(random_y, label="random")
    plt.legend(loc="upper left")
    plt.show()
