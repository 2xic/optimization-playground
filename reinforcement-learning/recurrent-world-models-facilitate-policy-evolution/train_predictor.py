from utils import *
from replay_buffer import ReplayBuffer
from optimization_playground_shared.rl.atari_env import Env
import torch.nn as nn
from train_encoder import Vae, get_trained_encoder
import random

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
            dim=-1
        )
        # If there is a single batch
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        assert len(x.shape) == 3, "Shape " + str(x.shape)
        assert len(hidden.shape) == 3, "Shape " + str(hidden.shape)
        
        x, hn = self.rnn(x, hidden)
        x = x.squeeze(0)
        x = self.model(x)
        return x, hn

def train_predictor(encoder: Vae):
    env = Env()
    model = Rnn(
        ACTION_SIZE=env.action_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    loss_predictor = LossTracker("predictor/loss")

    #load_model(model, optimizer, './predictor/model')

    progress = tqdm(range(PREDICTOR_TRAINING_STEPS), desc='Training rnn')
    max_size = 256

    index_replay_encoded_previous_image_replay_buffer = []
    index_replay_encoded_next_image_replay_buffer = [ ] 
    index_replay_action_replay_buffer = []
    for epoch in progress:
        need_refill = lambda: len(index_replay_action_replay_buffer) == 0 or not index_replay_action_replay_buffer[0].is_filled

        if not need_refill() and random.randint(0, 512) == 4:
            index_replay_encoded_previous_image_replay_buffer = []
            index_replay_encoded_next_image_replay_buffer = [ ] 
            index_replay_action_replay_buffer = []   

        while need_refill():
            for index, (observation, previous_observation, action) in enumerate(env.random_play()):
                if len(index_replay_encoded_previous_image_replay_buffer) <= index:
                    index_replay_encoded_previous_image_replay_buffer.append(ReplayBuffer(overflow=False, max_size=max_size))
                    index_replay_encoded_next_image_replay_buffer.append(ReplayBuffer(overflow=False, max_size=max_size))
                    index_replay_action_replay_buffer.append(ReplayBuffer(overflow=False, max_size=max_size))

                encoded_previous_image_replay_buffer = index_replay_encoded_previous_image_replay_buffer[index]
                encoded_next_image_replay_buffer = index_replay_encoded_next_image_replay_buffer[index]
                action_replay_buffer = index_replay_action_replay_buffer[index]

                observation = observation.unsqueeze(0)
                previous_observation = previous_observation.unsqueeze(0)
                with torch.no_grad():
                    encoded_previous_image_replay_buffer.add(
                        encoder.encode(previous_observation).detach()
                    )
                    encoded_next_image_replay_buffer.add(
                        encoder.encode(observation).detach()
                    )

                action_tensor = torch.zeros(1, env.action_size).to(DEVICE)
                action_tensor[0][action] = 1
                action_replay_buffer.add(action_tensor)
            progress.set_description(f'Size {index_replay_action_replay_buffer[0].items.shape}')

        hidden = None
        sum_loss = 0
        for (
            previous_image_iter,
            next_image_iter,
            action_iter,
        ) in zip(index_replay_encoded_previous_image_replay_buffer, index_replay_encoded_next_image_replay_buffer, index_replay_action_replay_buffer ):                
            for index, (
                previous_image,
                next_image,
                action
            ) in enumerate(zip(
                previous_image_iter,
                next_image_iter,
                action_iter
            )):
                if hidden is None:
                    hidden =  model.initial_state(batch_size=previous_image_iter.items.shape[0])
                    loss = 0

                assert index == 0
                (output, hidden) = model.forward(
                    previous_image.unsqueeze(0),
                    action.unsqueeze(0),
                    hidden
                )
                hidden = hidden.detach()
                optimizer.zero_grad()
                loss = torch.nn.MSELoss(reduction='sum')(output, next_image)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()

                if epoch % 10 == 0:
                    save_image(encoder.decode(output)[:1],
                            './predictor/predicted_next_frame_decoded.png')
                    save_image(encoder.decode(next_image)[:1],
                            './predictor/truth_next_frame_decoded.png')
                    save_image(encoder.decode(previous_image)[:1],
                            './predictor/truth_previous_frame_decoded.png')
                    save_model(model, optimizer, './predictor/model')

        progress.set_description(f'RNN loss {sum_loss}')
        loss_predictor.add_loss(sum_loss)
        loss_predictor.save()
    return model


def get_trained_predictor(env):
    model = Rnn(
        ACTION_SIZE=env.action_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    load_model(model, optimizer, './predictor/model')
    return model

def preview_predictions(encoder):
    env = Env()

    model = get_trained_predictor(env)

    hidden = None
    for _, (observation, previous_observation, action) in enumerate(env.random_play()):
        if hidden is None:
            hidden =  model.initial_state(batch_size=1)

        previous_image = encoder.encode(previous_observation.unsqueeze(0)).detach()
        next_image = encoder.encode(observation.unsqueeze(0)).detach()

        action_tensor = torch.zeros(1, env.action_size).to(DEVICE)
        action_tensor[0][action] = 1
        (output, hidden) = model.forward(
            previous_image,
            action_tensor,
            hidden
        )
        save_image(encoder.decode(output)[:1],
                './predictor/predicted_next_frame_decoded.png')
        save_image(encoder.decode(next_image)[:1],
                './predictor/truth_next_frame_decoded.png')
        save_image(encoder.decode(previous_image)[:1],
                './predictor/truth_previous_frame_decoded.png')

if __name__ == "__main__":
    encoder = get_trained_encoder()
    preview_predictions(encoder)


