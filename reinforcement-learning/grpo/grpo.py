# [GRPO](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo)
# https://unsloth.ai/blog/grpo
# https://github.com/lsdefine/simple_GRPO
# https://yugeten.github.io/posts/2025/01/ppogrpo/
# https://huggingface.co/blog/NormalUhr/grpo
# https://arxiv.org/pdf/2402.03300

from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch

class Agent:
    def __init__(self, state_size, action) -> None:
        self.model = nn.Sequential(*[
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, action),
            nn.Softmax(dim=1),
        ])
        self.old_model_parameter =  nn.Sequential(*[
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, action),
            nn.Sigmoid(),
            nn.LogSoftmax(dim=1)
            #nn.Softmax(dim=1),
        ])
        self.optimizer =  torch.optim.Adam(
            self.model.parameters(),
        )
        self.is_training = True
        self.accumulated_reward = 0

    def eval(self):
        self.is_training = False
        return self
    
    def forward(self, env: TicTacToe):
        state_actions = []
        while not env.done:
            assert env.player == 1
            state = torch.from_numpy(env.state).reshape((1, -1)).float()
            distro = self.model(state).log()
            action = None
            distro = torch.distributions.Categorical(distro)
            while action not in env.legal_actions:
                action = distro.sample()
            env.play(action)
            reward = env.winner
            next_state = torch.from_numpy(env.state).reshape((1, -1)).float()
            state_actions.append(
                (
                    state, action, 
                    (
                        -1 if reward is None else reward * 10
                    ),
                    distro.log_prob(action),
                    next_state,
                    env.done
                )
            )  
        return state_actions
    
    def train(self, env: TicTacToe):
        state_actions = self.forward(env)
        env_reward = env.winner if env.winner else 0
        
        loss = 0
        if self.is_training:
            gamma = 0.998
            eps = 0.2
            self.optimizer.zero_grad()
            rewards = torch.tensor([
                i[2] for i in state_actions
            ]).float()
            std_rewards = torch.std(rewards, dim=0)
            mean_rewards = torch.mean(rewards, dim=0)

            for index, (state, action, reward, _, _, _) in enumerate(state_actions):
                future_reward_sum = 0
                for iii in range(index, len(state_actions)):
                    future_reward_sum += gamma ** (iii - index - 1) * state_actions[iii][2]
                
                ppo_ratio = self.model(state) / self.old_model_parameter(state)
                A_i = (reward - mean_rewards) / (std_rewards + 1)
                
                """
                PPO-CLIP objective
                """
                a = ppo_ratio[0][action] * A_i
                b = torch.clip(
                    self.model(state) / self.old_model_parameter(state),
                    1 - eps,
                    1 + eps 
                ) * A_i
                loss += (torch.min(
                    a,
                    b
                ) * -1).mean()


            scaled_loss = loss * 1 / len(state_actions)
            assert not torch.isnan(scaled_loss)
            (scaled_loss).backward()
            # update model so that objective = old 
            self.old_model_parameter.load_state_dict(self.model.state_dict())        
            self.optimizer.step()
        self.accumulated_reward += env_reward
        return loss

if __name__ == "__main__":
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = Agent(4*4, env.action_space)
    epochs = 30_000

    agent_y = []
    for epochs in range(epochs):
        loss = agent.train(env)
        env.reset()

        agent_y.append(agent.accumulated_reward)

        if epochs % 1_000 == 0:
            print(epochs, loss, agent.accumulated_reward)

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
    plt.savefig('plot.png')
