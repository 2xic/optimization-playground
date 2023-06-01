from model import RewardModel
from collections import namedtuple
import torch
import random

class Rnd:
    def __init__(self, env, disable_rnd=False):
        self.N = 1_00
        self.N_opt = 1
        self.ROLLOUTS = 1_0
        self.WARM_UP_STEPS = 1_000
        self.t = 0
        self.env = env
        self.disable_rnd = disable_rnd
        
        self.freezed_model = RewardModel(env.action_space)
        self.train_model = RewardModel(env.action_space)

    def train(self, model):
        model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        reward_model_optimizer = torch.optim.Adam(self.train_model.parameters(), lr=1e-4)

        if not self.disable_rnd:
            for _ in range(self.WARM_UP_STEPS):
                while not self.env.done and self.env.winner is None:
                    actions = self.env.legal_actions
                    action = actions[random.randint(0, len(actions) - 1)]
                    (state, reward, action, gamma) = self.env.step(action)
                    self.t += 1
                self.env.reset()            
        
        self.rewards = []
        for i in range(self.N):
            entry = namedtuple('entry', 'r_i r_e state action')
            batch = []
            score = 0
            for _ in range(self.ROLLOUTS):
                while not self.env.done and self.env.winner is None:
                    env_state = self.env.env.float()
                    action = self.get_legal_actions(model, env_state)
                    (state, reward, action, gamma) = self.env.step(action)
                    exploration_reward = ((self.forward_freezed(env_state) - self.train_model(env_state)) ** 2).sum().item()
                    score += reward
                    batch.append(
                        entry(
                            r_e=reward,
                            r_i=exploration_reward,
                            state=state,
                            action=action
                        )
                    )
                self.env.reset()
            
            for _ in range(self.N_opt):
                batch = random.sample(batch, min(len(batch), 256))

                sum_model_loss = 0
                sum_reward_loss = 0
                model_optimizer.zero_grad()
                reward_model_optimizer.zero_grad()
                for entry in batch:

                    action = entry.action
                    reward = None
                    if self.disable_rnd:
                        reward = entry.r_e
                    else:
                        reward = entry.r_e + entry.r_i
                    state = entry.state.float()

                    prediction = model(state)
                    with torch.no_grad():
                        predicted_state = model(state).detach()
                        predicted_state[0][action] = reward

                    loss = torch.nn.MSELoss()(predicted_state, prediction)
                    sum_model_loss += loss

                    exploration_loss = torch.nn.MSELoss()(self.forward_freezed(state), self.train_model(state))
                    sum_reward_loss += exploration_loss
                    
                (sum_model_loss / 256).backward()
                (sum_reward_loss / 256).backward()
                #  print(1)
                print(f"{i} / {self.N} model={sum_model_loss}, reward={sum_reward_loss}, score={score}")
                reward_model_optimizer.step()
                model_optimizer.step()

            self.rewards.append(score)
        return self.rewards

    def forward_freezed(self, state):
        with torch.no_grad():
            return self.freezed_model(state)

    def get_legal_actions(self, model, state):
        model_output = model(state)
        for i in range(0, self.env.action_space):
            if i not in self.env.legal_actions:
                model_output[0][i] = 0
        action = torch.argmax(model_output[0])
        return action

