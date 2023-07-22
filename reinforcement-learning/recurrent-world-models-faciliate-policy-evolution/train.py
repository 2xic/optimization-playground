from env import Env
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch.optim as optim
from torchvision.utils import save_image
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn
from tqdm import tqdm
from optimization_playground_shared.plot.Plot import Plot, Figure


DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAINING_STEPS = 10_000
VAE_TRAINING_STEPS = TRAINING_STEPS #1_000  # 10_000
OTHER_TRAINING_STEPS = TRAINING_STEPS  #1_000  # 10_000

class LossTracker:
    def __init__(self, name):
        self.loss = []
        self.name = name

    def add_loss(self, item):
        self.loss.append(item)

    def save(self):
        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Loss": self.loss,
                    },
                    title="Loss",
                    x_axes_text="Epochs",
                    y_axes_text="Loss",
                    x_scale='symlog'
                )
            ],
            name="loss_" + self.name + '.png'
        )

class RewardTracker:
    def __init__(self, name):
        self.reward = []
        self.name = name

    def add_reward(self, item):
        self.reward.append(item)

    def save(self):
        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Reward": self.reward,
                    },
                    title="Reward",
                    x_axes_text="Epochs",
                    y_axes_text="Reward",
                    x_scale='symlog'
                )
            ],
            name="reward_" + self.name + '.png'
        )


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
            input_shape=(1, 160, 160),
            conv_shape=[
                32,
                64,
                128,
                256,
                512,
            ]
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.vae.parameters())

    def encode(self, observation):
        observation = observation.to(DEVICE)
        (mean, log_var) = self.vae.encode(observation)
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var*epsilon
        return z

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
        out = self.decode(out)
        loss = torch.nn.MSELoss(reduction='sum')(out, observation)
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


def train_predictor(encoder: Vae):
    env = Env()
    model = Rnn(
        ACTION_SIZE=env.action_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    loss_predictor = LossTracker("predictor")

    progress = tqdm(range(OTHER_TRAINING_STEPS), desc='Training rnn')
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
            loss += torch.nn.MSELoss(reduction='sum')(output, expected)
        loss.backward()
        optimizer.step()
        progress.set_description(f'RNN loss {loss.item()}')
        loss_predictor.add_loss(loss.item())
        loss_predictor.save()
    return model


def train_encoder():
    env = Env()
    encoder = Vae()
    replay_buffer = ReplayBuffer()
    loss_encoder = LossTracker("encoder")

    progress = tqdm(range(VAE_TRAINING_STEPS), desc='Training encoder')
    for epoch in progress:
        for (observation, _, _) in env.random_play():
            observation = observation.unsqueeze(0)
            replay_buffer.add(observation)
        (loss, out) = encoder.loss(replay_buffer.items)
        progress.set_description(f'Encoder loss {loss.item()}')

        if epoch % 10 == 0:
            save_image(out[:4], 'vae.png')
            save_image(replay_buffer.items[:4], 'truth.png')
        #    exit(0)
        loss_encoder.add_loss(loss.item())
        loss_encoder.save()
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
 #       print(action)
 #       print(torch.argmax(
 #           action,
 #           dim=1
 #       ))
        action = torch.argmax(
            action,
            dim=1
        )[0]
#        print(action)
#        print(action)
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
    loss_controller = LossTracker("controller")
    reward_over_time = RewardTracker("reward_controller")

    progress = tqdm(range(OTHER_TRAINING_STEPS), desc='Training agent')
    for _ in progress:

        agent.controller.zero_grad()
        loss = 0
        sum_reward = 0
        for (observation, action, reward, info) in env.agent_play(agent):
            (predicted, _) = agent.forward(observation, info)
            cloned = predicted.clone().detach()
            cloned[0][action] = reward

            #loss += ((cloned - predicted) ** 2).mean()
            loss += torch.nn.MSELoss(reduction='sum')(cloned, predicted)
            sum_reward += reward
        loss.backward()
        optimizer.step()
        progress.set_description(
            f'Agent loss {loss.item()} Reward: {sum_reward}')
        loss_controller.add_loss(loss.item())
        reward_over_time.add_reward(sum_reward)
        loss_controller.save()
        reward_over_time.save()

    return agent


if __name__ == "__main__":
    env = Env().reset()
   # print(env.shape)
   # save_image(env, 'env.png')
   # exit(0)

    encoder = train_encoder()
    rnn = train_predictor(encoder)
    agent = train_controller(
        rnn,
        encoder,
    )
