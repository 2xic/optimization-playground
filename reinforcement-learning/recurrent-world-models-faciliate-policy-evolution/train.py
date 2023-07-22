from env import Env
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch.optim as optim
from torchvision.utils import save_image as torch_save_image
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn
from tqdm import tqdm
from optimization_playground_shared.plot.Plot import Plot, Figure
import os
import json
import random
from optimization_playground_shared.rl.epsilon import Epsilon

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAINING_STEPS = 1_000
VAE_TRAINING_STEPS = 0 #TRAINING_STEPS
PREDICTOR_TRAINING_STEPS = 0 #TRAINING_STEPS
CONTROLLER_TRAINING_STEPS = TRAINING_STEPS


def save_model(model_class, optimizer_class, name):
    print(f"Saving model {name}")
    torch.save({
        'model_state_dict': model_class.state_dict(),
        'optimizer_state_dict': optimizer_class.state_dict(),
    }, name)


def load_model(model_class, optimizer_class, name):
    print(f"Loading model {name}")
    checkpoint = torch.load(name)
    model_class.load_state_dict(checkpoint['model_state_dict'])
    optimizer_class.load_state_dict(checkpoint['optimizer_state_dict'])


def save_image(tensor, file):
    dir = os.path.dirname(os.path.abspath(file))
    os.makedirs(dir, exist_ok=True)
    torch_save_image(tensor, file)


class LossTracker:
    def __init__(self, name):
        self.loss = []
        self.name = name

    def add_loss(self, item):
        self.loss.append(item)

    def save(self):
        file = self.name + '.png'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

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
                    #x_scale='symlog'
                )
            ],
            name=file
        )


class DebugInfo:
    def __init__(self, name):
        self.debug = []
        self.name = name

    def add_debug_info(self, **kwargs):
        self.debug.append(kwargs)

    def save(self):
        file = self.name + '.json'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

        with open(file, "w") as file:
            file.write(json.dumps(self.debug, indent=4))


class RunningAverage:
    def __init__(self) -> None:
        self.Q = 0
        self.N = 0

    def update(self, value):
        self.Q = self.Q + (value - self.Q) / (self.N + 1)
        self.N += 1
        return self.Q

class RewardTracker:
    def __init__(self, name):
        self.reward = []
        self.reference_reward = []
        self.name = name
        self.reward_avg = RunningAverage()
        self.reference_reward_avg = RunningAverage()

    def add_reward(self, item):
        self.reward.append(self.reward_avg.update(item))

    def add_reference_reward(self, item):
        self.reference_reward.append(self.reference_reward_avg.update(item))

    def save(self):
        file = self.name + '.png'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Reward": (self.reward),
                        "Reference reward (untrained)": (self.reference_reward),
                    },
                    title="Reward",
                    x_axes_text="Epochs",
                    y_axes_text="Reward",
                )
            ],
            name=file
        )


class Rnn(nn.Module):
    def __init__(self, ACTION_SIZE=1, Z_SHAPE=128):
        super().__init__()
        # input, output, hidden
        self.OUTPUT_SHAPE = 120
        self.ACTION_SIZE = ACTION_SIZE
        self.HIDDEN_SIZE = 2
        self.rnn = nn.RNN(Z_SHAPE + ACTION_SIZE,
                          self.OUTPUT_SHAPE, self.HIDDEN_SIZE)
        self.model = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.Linear(self.OUTPUT_SHAPE, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, Z_SHAPE)
        ])

    def initial_state(self, batch_size):
        return torch.randn(self.HIDDEN_SIZE, batch_size, self.OUTPUT_SHAPE).to(DEVICE)

    def forward(self, previous_state, action, hidden):
        x = torch.concat(
            (previous_state, action),
            dim=1
        ).unsqueeze(1)
        assert len(x.shape) == 3

        x, hn = self.rnn(x, hidden)
        x = x.reshape((x.shape[0], -1))
        x = self.model(x)
        return x, hn


class Controller(nn.Module):
    def __init__(self, ACTION_SIZE, Z_SHAPE=128):
        super().__init__()
        HIDDEN_SHAPE = 2 * 120
        self.model = nn.Sequential(*[
            nn.Linear(Z_SHAPE + HIDDEN_SHAPE, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, ACTION_SIZE),
            nn.Sigmoid(),
        ]).to(DEVICE)

    def forward(self, Z, hidden):
        assert len(Z.shape) == 2, "Output should be resized before inputted"
        x = torch.concat(
            (Z, hidden),
            dim=1
        )
        x = self.model(x)
        return x

class Vae:
    def __init__(self) -> None:
        self.vae = SimpleVaeModel(
            input_shape=(1, 40, 40),
            conv_shape=[
                32,
                64,
                128,
            ],
            z_size=128,
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

    def loss(self, observation):
        self.vae.zero_grad()

        observation = observation.to(DEVICE)
        out = self.encode(observation)
        out = self.decode(out)
        loss = torch.nn.MSELoss(reduction='sum')(out, observation)
        loss.backward()

        self.optimizer.step()

        return loss, out


def train_predictor(encoder: Vae):
    env = Env()
    model = Rnn(
        ACTION_SIZE=env.action_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    loss_predictor = LossTracker("predictor/loss")

    load_model(model, optimizer, './predictor/model')

    progress = tqdm(range(PREDICTOR_TRAINING_STEPS), desc='Training rnn')
    for epoch in progress:
        encoded_previous_image_replay_buffer = ReplayBuffer(sequential=True)
        encoded_next_image_replay_buffer = ReplayBuffer(sequential=True)
        action_replay_buffer = ReplayBuffer(sequential=True)

        for (observation, previous_observation, action) in env.random_play():
            observation = observation.unsqueeze(0)
            previous_observation = previous_observation.unsqueeze(0)
            encoded_previous_image_replay_buffer.add(
                encoder.encode(previous_observation).detach()
            )
            encoded_next_image_replay_buffer.add(
                encoder.encode(observation).detach()
            )

            action_tensor = torch.zeros(1, env.action_size).to(DEVICE)
            action_tensor[0][action] = 1
            action_replay_buffer.add(
                action_tensor
            )

        optimizer.zero_grad()
        hidden = model.initial_state(batch_size=1)
        loss = 0
        for index in range(encoded_next_image_replay_buffer.items.shape[0]):
            input_frame = encoded_previous_image_replay_buffer.items[index].unsqueeze(
                0)
            (output, hidden) = model.forward(
                input_frame,
                action_replay_buffer.items[index].unsqueeze(0),
                hidden
            )
            expected = encoded_next_image_replay_buffer.items[index].unsqueeze(
                0)
            loss += torch.nn.MSELoss(reduction='sum')(output, expected)

        if epoch % 10 == 0:
            save_image(encoder.decode(output),
                       './predictor/predicted_next_frame_decoded.png')
            save_image(encoder.decode(expected),
                       './predictor/truth_next_frame_decoded.png')
            save_image(encoder.decode(input_frame),
                       './predictor/truth_previous_frame_decoded.png')
            save_model(model, optimizer, './predictor/model')

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
    # many more items !!!
    replay_buffer.max_size = 4096
    loss_encoder = LossTracker("./encoder/loss")

    load_model(encoder.vae, encoder.optimizer, './encoder/model')

    progress = tqdm(range(VAE_TRAINING_STEPS), desc='Training encoder')
    for epoch in progress:
        if not replay_buffer.is_filled or random.randint(0, 10) == 2:
            for (observation, _, _) in env.random_play():
                observation = observation.unsqueeze(0)
                replay_buffer.add(observation)

        if replay_buffer.is_filled:
            for batch in replay_buffer:
                (loss, out) = encoder.loss(batch)
            progress.set_description(f'Encoder loss {loss.item()}')

            if epoch % 10 == 0:
                save_image(out[:4], './encoder/vae.png')
                save_image(replay_buffer.items[:4], './encoder/truth.png')
                save_model(encoder.vae, encoder.optimizer, './encoder/model')

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
        self.explorer = Epsilon(decay=0.9999)

    def reset(self):
        self.h = self.rnn.initial_state(batch_size=1)

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
#    reward_over_time_untrained = RewardTracker("./controller/reward_over_time_untrained")
    debug_info = DebugInfo('./controller/debug')

    observation_replay_buffer = ReplayBuffer()
    observation_replay_buffer.max_size = 4096
    h_replay_buffer = ReplayBuffer()
    h_replay_buffer.max_size = 4096
    action_replay_buffer = ReplayBuffer()
    action_replay_buffer.max_size = 4096
    reward_replay_buffer = ReplayBuffer()
    reward_replay_buffer.max_size = 4096

    progress = tqdm(range(CONTROLLER_TRAINING_STEPS), desc='Training agent')
    for epoch in progress:
        loss = 0
        sum_reward = 0
        sum_reward_untrained = 0
        counter = 0

        with torch.no_grad():
            for (_, _, reward, _) in env.agent_play(untrained_agent):
                sum_reward_untrained += reward

        for (observation, action, reward, old_h) in env.agent_play(agent):
            observation_replay_buffer.add(observation.unsqueeze(0))
            h_replay_buffer.add(old_h.reshape((1, -1)))

            action_tensor = torch.zeros(1).to(DEVICE).long()
            action_tensor[0] = action

            action_replay_buffer.add(action_tensor)
            sum_reward += reward
            counter += 1

#        sum_reward /= 20
        decay = 0.98
        for index in range(counter):
            if index + 3 >= counter:
                reward_replay_buffer.add(torch.tensor([
                    0
                ], device=DEVICE))
            else:
                reward_replay_buffer.add(torch.tensor([
                    ((sum_reward / 20) * (decay ** (index)))
                ], device=DEVICE))

        for (observation, old_h, action, reward) in zip(
            list(iter(observation_replay_buffer)),
            list(iter(h_replay_buffer)),
            list(iter(action_replay_buffer)),
            list(iter(reward_replay_buffer)),
        ):
            optimizer.zero_grad()
            predicted = agent._train_get_actions(observation, old_h)
            cloned = predicted.clone().detach()
            #cloned[:, action] = reward
            cloned[torch.arange(len(cloned)), action] = reward

            loss = torch.nn.MSELoss(reduction='mean')(cloned, predicted)
            
            loss.backward()
            optimizer.step()

            epsilon = agent.explorer.epsilon
            progress.set_description(
                f'Agent loss {loss.item()} Reward: {sum_reward}, Epsilon: {epsilon}'
            )
        if epoch % 10 == 0:
            save_model(agent.controller, optimizer, './controller/model')

        loss_controller.add_loss(loss.item())
        reward_over_time.add_reward(sum_reward)
        reward_over_time.add_reference_reward(sum_reward_untrained)
        loss_controller.save()
        reward_over_time.save()
        debug_info.save()

    return agent


if __name__ == "__main__":
    encoder = train_encoder()
    rnn = train_predictor(encoder)
    agent = train_controller(
        rnn,
        encoder,
    )
