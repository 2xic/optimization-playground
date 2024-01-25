from config import Config
from models.torch_model import Model # TODO: Should use tinygrad actually
from mcts import MonteCarloSearchTree
from torch import Tensor
from replay_buffer import ReplayBuffer, Predictions
from debug import Debug
import torch
from loss_debug import LossDebug
import torch.nn as nn

class Agent:
    def __init__(self, config: Config, env) -> None:
        self.config = config
        self.model = Model(self.config)        
        self.opt = self.model.get_optimizer()
        
        # TODO: Replay buffer
        self.env = env
        self.replay_buffer = ReplayBuffer()
        self.loss_debug = LossDebug()

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
            print(old_state, action, reward)
            assert reward in [0, 1]
            
            self.replay_buffer.add_to_entries(Predictions(
                state=old_state,
                action=action,
                environment_reward=reward,
                next_state=self.env.state.copy(),
                state_distribution=self.mcts.root.visited_probabilities
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
        protection_loss_ratio = 0 # 0.3
        policy_loss_ratio = 0.5
        prediction_reward_ratio = 1

        error = 0
        sum_reward_loss = 0
        sum_predicted_policy_loss = 0

        for i in self.replay_buffer.entries:
            state = Tensor(i.state)
            # The predicted next state 
            encoded_state = self.model.encode_state(state)
            next_state = self.model.get_next_state(
                encoded_state,
                int(i.action)
            )
            reward = self.model.get_state_reward(
                next_state
            )
            environment_reward = Tensor([i.environment_reward])

            # Policy loss
            # predicted mcts vs 
            predicted_policy = self.model.get_policy_predictions(
                encoded_state
            )
            # todo: cross entropy ? 
            predicted_policy_loss = nn.CrossEntropyLoss()(
                predicted_policy.reshape((1, -1)).float(),
                i.state_distribution.reshape((1, -1)).float(),
            )
            assert torch.allclose(torch.sum(predicted_policy.reshape((1, -1)).float()), torch.tensor([1]).float())
            assert torch.allclose(torch.sum(i.state_distribution.reshape((1, -1)).float()), torch.tensor([1]).float())

            # Reward loss
            assert environment_reward.item() in [0, 1], environment_reward.item()
            assert reward.item() < 1 and reward.item() > -1
            reward_loss = ((reward.reshape(-1) - environment_reward) ** 2).sum(axis=0)

            # Self consistency loss
            # TODO: I don't think this is the best way to do this
            # Note that is is meant to be similar to https://arxiv.org/pdf/2011.10566.pdf
            prediction = self.model.get_state_prediction(next_state)

            encoded_next_state = self.model.encode_state(Tensor(i.next_state))
            projector_loss_real = self.model.get_state_projection(encoded_next_state)

            projection_loss = 0# (prediction - projector_loss_real).reshape((-1)).sum(axis=0).float() ** 2

            small_error = reward_loss * prediction_reward_ratio + \
                    predicted_policy_loss * policy_loss_ratio #+ \
                    #projection_loss * protection_loss_ratio 
            # TODO: 
            #print([
            #    reward_loss,
            #    projection_loss,
            #    predicted_policy_loss,
            #])
            sum_reward_loss += reward_loss.item()
            sum_predicted_policy_loss += predicted_policy_loss.item()

            if torch.isfinite(small_error).item():
                #print(f"small_error === {small_error}")
                #print(f"\t{reward_loss}")
                #print(f"\t{projection_loss}")
                #print(f"\t{predicted_policy_loss}")
                small_error.backward()
                error += small_error.item()
        self.loss_debug.add(
            reward=sum_reward_loss,
            policy=sum_predicted_policy_loss,
        )
        self.opt.step()
        loss = error
        return loss
