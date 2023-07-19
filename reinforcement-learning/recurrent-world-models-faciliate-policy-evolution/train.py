from env import Env
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn


class Rnn(nn.Module):
    def __init__(self):
        super().__init__()
        # input, output, hidden
        self.OUTPUT_SHAPE = 120
        self.rnn = nn.RNN(513, self.OUTPUT_SHAPE, 2)
        self.linear = nn.Linear(
            self.OUTPUT_SHAPE,
            512
        )

    def initial_state(self):
        return torch.randn(2, 1, self.OUTPUT_SHAPE)

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
    def __init__(self):
        super().__init__()
        Z_SHAPE = 512
        HIDDEN_SHAPE = 2 * 120
        self.model = nn.Sequential(*[
            nn.Linear(Z_SHAPE + HIDDEN_SHAPE, 2)
        ])

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
        )
        self.optimizer = optim.Adam(self.vae.parameters())

    def encode(self, observation):
        return self.vae.encode(observation)

    def decode(self, observation):
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

        out = self.vae.encode(observation)
        out = self.vae.decode(
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
        self.vae = Vae(

        )

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
    previous_image_replay_buffer = ReplayBuffer(sequential=True)
    next_image_replay_buffer = ReplayBuffer(sequential=True)
    action_replay_buffer = ReplayBuffer(sequential=True)
    for (observation, previous_observation, action) in env.random_play():
        observation = observation.unsqueeze(0)
        previous_observation = previous_observation.unsqueeze(0)
        previous_image_replay_buffer.add(
            encoder.encode(previous_observation)
        )
        next_image_replay_buffer.add(
            encoder.encode(observation)
        )
        action_replay_buffer.add(
            torch.tensor([action]).unsqueeze(0)
        )
    model = Rnn()
    (output, hidden) = model.forward(
        previous_image_replay_buffer.items[:1],
        action_replay_buffer.items[:1],
        model.initial_state()
    )
    print(output.shape)
    print(next_image_replay_buffer.items.shape)

    controller = Controller()
    a = controller.forward(
        previous_image_replay_buffer.items[:1],
        hidden
    )
    print(a.shape)

def train_encoder():
    env = Env()
    encoder = Vae()
    replay_buffer = ReplayBuffer()
    # observation = env.reset()
    for epoch in range(10_000):
        for (observation, _, _) in env.random_play():
            observation = observation.unsqueeze(0)
            replay_buffer.add(observation)
        (loss, out) = encoder.loss(replay_buffer.items)
        print(epoch)
        if epoch % 10 == 0:
            print(loss)
            save_image(out, 'vae.png')
            save_image(replay_buffer.items, 'truth.png')
        break
    return encoder


if __name__ == "__main__":
    encoder = train_encoder()
    train_predictor(encoder)
