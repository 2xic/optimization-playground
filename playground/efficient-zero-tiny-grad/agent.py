from config import Config
from model import Model
from mcts import MonteCarloSearchTree
from tinygrad import nn, Tensor
from replay_buffer import ReplayBuffer, Predictions
from debug import Debug
import json

class Agent:
    def __init__(self, config: Config, env) -> None:
        self.config = config
        self.model = Model(self.config)
        parameters = nn.state.get_parameters(self.model)
       # print(parameters)
        self.opt = nn.optim.Adam(parameters, lr=3e-4)
        # TODO: Replay buffer
        self.env = env
        self.replay_buffer = ReplayBuffer()

    def play(self, debugger: Debug):
        self.env.reset()
        terminated = False
        sum_reward = 0
        while not terminated:
            # Play one epoch 
            old_state = self.env.state.copy()
            self.mcts = MonteCarloSearchTree.from_state(
                self.env.state,
                self.model,
                self.config,
            )
            self.mcts.expand()

            action = self.mcts.get_action(self.mcts.root)
            (
                _,
                reward,
                terminated,
                _,
                _
            ) = self.env.step(action)
            self.replay_buffer.add_to_entries(Predictions(
                state=old_state,
                action=action,
                environment_reward=reward,
                next_state=self.env.state.copy(),
            ))
            debugger.store_predictions(
                self.model.get_state_reward(self.model.encode_state(Tensor(self.env.state))),
                self.model.encode_state(Tensor(self.env.state)),
            )
            sum_reward += reward
        return sum_reward
        

    def loss(self):
        self.opt.zero_grad()
        loss = 0

        # Ratios
        protection_loss_ratio = 0.4
        prediction_reward_ratio = 1

        for i in self.replay_buffer.entries:
            state = Tensor(i.state)
            # The predicted next state 
            next_state = self.model.get_next_state(
                self.model.encode_state(state),
                int(i.action)
            )
            reward = self.model.get_state_reward(
                next_state
            )
            env_reward = Tensor(i.environment_reward)

            # Reward loss
            last_state_projector = next_state.sequential(self.model.projector_network).sigmoid()
            prediction = last_state_projector.sequential(self.model.predictor).sigmoid()
            reward_loss = ((reward.reshape(-1) - env_reward) ** 2).sum(axis=0)

            # Self consistency loss
            # TODO: I don't think this is the best way to do this
            # Note that is is meant to be similar to https://arxiv.org/pdf/2011.10566.pdf
            encoded_next_state = self.model.encode_state(Tensor(i.next_state))
            projector_loss_real = encoded_next_state.sequential(self.model.projector_network).sigmoid()

            projection_loss = (prediction - projector_loss_real).reshape((-1)).sum(axis=0).float() ** 2

            error = reward_loss * prediction_reward_ratio + \
                    projection_loss * protection_loss_ratio
            error.backward()
            
            loss += error.item()
            
        self.opt.step()
        return loss
