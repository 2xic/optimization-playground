from env import Env
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn
from tqdm import tqdm


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAINING_STEPS = 100 # 10_000


class Rnn(nn.Module):
    def __init__(self, ACTION_SIZE=1):
        super().__init__()
        # input, output, hidden
        self.OUTPUT_SHAPE = 120
        self.ACTION_SIZE = ACTION_SIZE
        self.rnn = nn.RNN(512 + ACTION_SIZE, self.OUTPUT_SHAPE, 2)
        self.linear = nn.Linear(
            self.OUTPUT_SHAPE,
            512
        )

    def initial_state(self):
        return torch.randn(2, 1, self.OUTPUT_SHAPE).to(DEVICE)

    def forward(self, previous_state, action, hidden):
        x = torch.concat(
            (previous_state, action),
            dim=1
        ).unsqueeze(1)
        assert len(x.shape) == 3

        x, hn = self.rnn(x, hidden)
        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)
        return x, hn


class Controller(nn.Module):
    def __init__(self, ACTION_SIZE):
        super().__init__()
        Z_SHAPE = 512
        HIDDEN_SHAPE = 2 * 120
        self.model = nn.Sequential(*[
            nn.Linear(Z_SHAPE + HIDDEN_SHAPE, ACTION_SIZE)
        ]).to(DEVICE)

    def forward(self, Z, hidden):
        x = torch.concat(
            (Z,
             hidden.reshape((Z.shape[0], -1)),),
            dim=1
        )
        x = self.model(x)
        return x

class Vae:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            input_shape=(3, 210, 160)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.vae.parameters())

    def encode(self, observation):
        observation = observation.to(DEVICE)
        return self.vae.encode(observation)

    def decode(self, observation):
        observation = observation.to(DEVICE)
        return self.vae.decode(observation)

    def save(self, name):
        torch.save({
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def loss(self, observation):
        self.vae.zero_grad()

        observation = observation.to(DEVICE)
        out = self.encode(observation)
        out = self.decode(
            out)[:, :3, :observation.shape[2], :observation.shape[3]]
        loss = ((observation - out) ** 2).sum()
        loss.backward()

        self.optimizer.step()

        return loss, out


class Rollout:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.rnn = Rnn()
        self.controller = None
        self.vae = Vae()

    def rollout(self):
        observation = self.env.reset()
        cumulative_reward = 0
        done = False

        h = self.rnn.initial_state()

        while not done:
            z = self.vae.encode(observation)  # vae encoding
            action = self.controller.action([z, h])
            observation, reward, done = self.env.step(action)

            cumulative_reward += reward
            h = self.rnn.forward([
                action,
                z,
                h,
            ])
        return cumulative_reward


def train():
    """
    1. Rollout 10_000 iterations to train VAE 
    11. Objective is learning to reconstruct the frame (we use the encoded output only later)
    2. Train the MDN-RNN 
    2.1 (predict next frame given previous + action + hidden state)
    3. Train the controller
    """
    pass


def train_predictor(encoder: Vae):
    env = Env()
    model = Rnn(
        ACTION_SIZE=env.action_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
            
    progress = tqdm(range(TRAINING_STEPS), desc='Training rnn')
    for _ in progress:
        previous_image_replay_buffer = ReplayBuffer(sequential=True)
        next_image_replay_buffer = ReplayBuffer(sequential=True)
        action_replay_buffer = ReplayBuffer(sequential=True)
        
        for (observation, previous_observation, action) in env.random_play():
            observation = observation.unsqueeze(0)
            previous_observation = previous_observation.unsqueeze(0)
            previous_image_replay_buffer.add(
                encoder.encode(previous_observation).detach()
            )
            next_image_replay_buffer.add(
                encoder.encode(observation).detach()
            )

            action_tensor = torch.zeros(1, env.action_size).to(DEVICE)
            action_tensor[0][action] = 1
            action_replay_buffer.add(
                action_tensor
            )

        optimizer.zero_grad()
        hidden = model.initial_state()
        loss = 0
        for index in range(next_image_replay_buffer.items.shape[0]):
            (output, hidden) = model.forward(
                previous_image_replay_buffer.items[index].unsqueeze(0),
                action_replay_buffer.items[index].unsqueeze(0),
                hidden
            )
            expected = next_image_replay_buffer.items[index].unsqueeze(0)

            loss += ((output - expected) ** 2).mean()
        loss.backward()
        optimizer.step()
        progress.set_description(f'RNN loss {loss.item()}')
    return model


def train_encoder():
    env = Env()
    encoder = Vae()
    replay_buffer = ReplayBuffer()
    # observation = env.reset()
    progress = tqdm(range(TRAINING_STEPS), desc='Training encoder')
    for epoch in progress:
        for (observation, _, _) in env.random_play():
            observation = observation.unsqueeze(0)
            replay_buffer.add(observation)
        (loss, out) = encoder.loss(replay_buffer.items)
        progress.set_description(f'Encoder loss {loss.item()}')

        if epoch % 10 == 0:
            save_image(out[:4], 'vae.png')
            save_image(replay_buffer.items[:4], 'truth.png')
    return encoder

class Agent:
    def __init__(self, rnn: Rnn, encoder: Vae) -> None:
        self.rnn = rnn
        self.encoder = encoder
        self.controller = Controller(
            ACTION_SIZE=self.rnn.ACTION_SIZE
        ).to(DEVICE)
        self.h = None

    def reset(self):
        self.h = self.rnn.initial_state()

    def action(self, observation):
        old_h = self.h.clone()
        action, new_h = self.forward(observation, self.h)
        action = torch.argmax(
            action,
            dim=1
        )[0]
        self.h = new_h
        return action, old_h
    
    def forward(self, observation, h):
        observation = observation.to(DEVICE)
        h = h.to(DEVICE)
        new_h = None
        state = None

        with torch.no_grad():
            state = self.encoder.encode(
                observation.unsqueeze(0)
            )
        action = self.controller.forward(
            state,
            h
        )
        with torch.no_grad():
            (_, new_h) = self.rnn.forward(
                state,
                action,
                h
            )
        return action, new_h

def train_controller(rnn: Rnn, encoder: Vae):
    env = Env()
    agent = Agent(
        rnn,
        encoder
    )
    optimizer = torch.optim.Adam(agent.controller.parameters())

    progress = tqdm(range(TRAINING_STEPS), desc='Training agent')
    for _ in progress:
        
        agent.controller.zero_grad()
        loss = 0
        sum_reward = 0
        for (observation, action, reward, info) in env.agent_play(agent):
            (predicted, _) = agent.forward(observation, info)
            cloned = predicted.clone().detach()
            cloned[0][action] = reward

            loss += ((cloned - predicted) ** 2).mean()
            reward += reward
        loss.backward()
        optimizer.step()
        progress.set_description(f'Agent loss {loss.item()} Reward: {sum_reward}')

    return agent

if __name__ == "__main__":
    encoder = train_encoder()
    rnn = train_predictor(encoder)
    agent = train_controller(
        rnn,
        encoder,
    )

