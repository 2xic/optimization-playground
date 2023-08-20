from env import Env
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn
from tqdm import tqdm
from optimization_playground_shared.rl.epsilon import Epsilon
from utils import *
from train_encoder import Vae, train_encoder
from train_predictor import Rnn, train_predictor
from collections import defaultdict

class Controller(nn.Module):
    def __init__(self, ACTION_SIZE, Z_SHAPE=128):
        super().__init__()
        HIDDEN_SHAPE = 2 * 120
        self.model = nn.Sequential(*[
            nn.Linear(Z_SHAPE + HIDDEN_SHAPE, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, ACTION_SIZE),
         #   nn.Sigmoid(),
        ]).to(DEVICE)

    def forward(self, Z, hidden):
        assert len(Z.shape) == 2, "Output should be resized before inputted"
        x = torch.concat(
            (Z, hidden),
            dim=1
        )
        x = self.model(x)
        return x



class Agent:
    def __init__(self, rnn: Rnn, encoder: Vae) -> None:
        self.rnn = rnn
        self.encoder = encoder
        self.controller = Controller(
            ACTION_SIZE=self.rnn.ACTION_SIZE
        ).to(DEVICE)
        self.h = None
        self.explorer = Epsilon(decay=0.9999)
        self.model_action_distribution = defaultdict(int)


    def reset(self):
        self.h = self.rnn.initial_state(batch_size=1)
        self.model_action_distribution = defaultdict(int)

    def action(self, observation):
        old_h = self.h.clone()
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        action, new_h = self.forward(observation, self.h)
        action = torch.argmax(
            action,
            dim=1
        )[0]
        self.h = new_h

        off_policy_action = self.explorer.action(list(range(self.rnn.ACTION_SIZE)))
        if off_policy_action:
            return off_policy_action, old_h

        self.model_action_distribution[action.item()] += 1
        return action, old_h

    def forward(self, observation, h):
        observation = observation.to(DEVICE)
        h = h.to(DEVICE)
        new_h = None
        state = None

        with torch.no_grad():
            state = self.encoder.encode(
                observation
            )
        # the batch size
        assert h.shape[1] == 1
        action = self.controller.forward(
            state,
            h.reshape((1, -1))
        )
        with torch.no_grad():
            (_, new_h) = self.rnn.forward(
                state,
                action,
                h
            )
        return action, new_h

    def _train_get_actions(self, observation, h):
        observation = observation.to(DEVICE)
        h = h.to(DEVICE)
        state = None

        with torch.no_grad():
            state = self.encoder.encode(
                observation
            )
        action = self.controller.forward(
            state,
            h
        )
        return action


class BestReplayBuffer:
    def __init__(self) -> None:
        self.observation_replay_buffer = ReplayBuffer()
        self.observation_replay_buffer.max_size = 4096
        self.h_replay_buffer = ReplayBuffer()
        self.h_replay_buffer.max_size = 4096
        self.action_replay_buffer = ReplayBuffer()
        self.action_replay_buffer.max_size = 4096
        self.reward_replay_buffer = ReplayBuffer()
        self.reward_replay_buffer.max_size = 4096

        self.avg_score = RunningAverage()

    def should_copy(self, reward):
        if self.avg_score.Q < reward:
            self.avg_score.update(reward)
            return True
        return False

def train_controller(rnn: Rnn, encoder: Vae):
    env = Env()

    untrained_agent = Agent(
        rnn,
        encoder
    )
    agent = Agent(
        rnn,
        encoder
    )
    optimizer = torch.optim.Adam(agent.controller.parameters())

    #load_model(agent.controller, optimizer, './controller/model')

    loss_controller = LossTracker("./controller/loss")
    reward_over_time = RewardTracker("./controller/reward")
    debug_info = DebugInfo('./controller/debug')

    observation_replay_buffer = ReplayBuffer()
    observation_replay_buffer.max_size = 512
    h_replay_buffer = ReplayBuffer()
    h_replay_buffer.max_size = 512
    action_replay_buffer = ReplayBuffer()
    action_replay_buffer.max_size = 512
    reward_replay_buffer = ReplayBuffer()
    reward_replay_buffer.max_size = 512

    best_replay_buffer = BestReplayBuffer()

    progress = tqdm(range(CONTROLLER_TRAINING_STEPS), desc='Training agent')
    for epoch in progress:
        action_distribution = defaultdict(int)
        loss = 0
        sum_reward = 0
        sum_reward_untrained = 0
        counter = 0

        with torch.no_grad():
            for (_, _, reward, _) in env.agent_play(untrained_agent):
                sum_reward_untrained += reward
        
        env_run = list(env.agent_play(agent))
        for (observation, action, reward, old_h) in env_run:
            observation_replay_buffer.add(observation.unsqueeze(0))
            h_replay_buffer.add(old_h.reshape((1, -1)))

            action_tensor = torch.zeros(1).to(DEVICE).long()
            action_tensor[0] = action

            action_replay_buffer.add(action_tensor)
            sum_reward += reward
            counter += 1
            action_distribution[action_tensor.item()] += 1

        decay = 0.98
        # If more than 50% of the action si a noop we set the reward to 0
        adjusted_reward = sum_reward
        for action_usage in action_distribution:
            if 0.5 < action_distribution[action_usage] / counter:
                adjusted_reward = -100
                break

        for index in range(counter):
            if index + 3 >= counter:
                # We failed
                reward_replay_buffer.add(torch.tensor([
                    -50
                ], device=DEVICE))
            else:
                reward_replay_buffer.add(torch.tensor([
                    (((adjusted_reward)) * (decay ** (counter - index)))
                ], device=DEVICE))

        """
        This is duplicate code...
        """
        if best_replay_buffer.should_copy(adjusted_reward):
            for (observation, action, reward, old_h) in env_run:
                best_replay_buffer.observation_replay_buffer.add(observation.unsqueeze(0))
                best_replay_buffer.h_replay_buffer.add(old_h.reshape((1, -1)))
                action_tensor = torch.zeros(1).to(DEVICE).long()
                action_tensor[0] = action
                best_replay_buffer.action_replay_buffer.add(action_tensor)
        decay = 0.98
        for index in range(counter):
            if index + 3 >= counter:
                # We failed
                best_replay_buffer.reward_replay_buffer.add(torch.tensor([
                    -50
                ], device=DEVICE))
            else:
                best_replay_buffer.reward_replay_buffer.add(torch.tensor([
                    (((adjusted_reward)) * (decay ** (counter - index)))
                ], device=DEVICE))

        """
        End duplicate code
        """
        overall_loss = 0
        for (observation, old_h, action, reward) in zip(
            list(iter(best_replay_buffer.observation_replay_buffer)) + list(iter(observation_replay_buffer)),
            list(iter(best_replay_buffer.h_replay_buffer)) + list(iter(h_replay_buffer)),
            list(iter(best_replay_buffer.action_replay_buffer)) + list(iter(action_replay_buffer)),
            list(iter(best_replay_buffer.reward_replay_buffer)) + list(iter(reward_replay_buffer)),
        ):
            predicted = agent._train_get_actions(observation, old_h)
            cloned = predicted.clone().detach()
            #cloned[:, action] = reward
            cloned[torch.arange(len(cloned)), action] = reward

            optimizer.zero_grad()
            loss = torch.nn.MSELoss(reduction='sum')(cloned, predicted)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

        epsilon = agent.explorer.epsilon
        best_replay_buffer_score = best_replay_buffer.avg_score.Q
        progress.set_description(
            f'Agent loss {overall_loss} Reward: {adjusted_reward}, Epsilon: {epsilon}, game length: {counter}'
        )
        if epoch % 10 == 0:
            save_model(agent.controller, optimizer, './controller/model')
            print(agent.model_action_distribution)
            print(action_distribution, counter, best_replay_buffer_score)

        loss_controller.add_loss(overall_loss)
        reward_over_time.add_reward(sum_reward)
        reward_over_time.add_reference_reward(sum_reward_untrained)
        loss_controller.save()
        reward_over_time.save()
        debug_info.save()

    return agent

def train_loop():
    encoder = train_encoder()
    rnn = train_predictor(encoder)
    agent = train_controller(
        rnn,
        encoder,
    )

if __name__ == "__main__":
    if False:
        train_loop()
