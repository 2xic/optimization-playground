from utils import *
from replay_buffer import ReplayBuffer
from optimization_playground_shared.rl.atari_env import Env
import random
import torch.optim as optim
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel

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

def train_encoder():
    env = Env()
    encoder = Vae()
    replay_buffer = ReplayBuffer()
    # many more items !!!
    replay_buffer.max_size = 4096
    loss_encoder = LossTracker("./encoder/loss")

    #load_model(encoder.vae, encoder.optimizer, './encoder/model')

    progress = tqdm(range(VAE_TRAINING_STEPS), desc='Training encoder')
    for epoch in progress:
        while not replay_buffer.is_filled or random.randint(0, 10) == 2:
            for (observation, _, _, _) in env.random_play():
                observation = observation.unsqueeze(0)
                replay_buffer.add(observation)

        sum_loss = 0
        for batch in replay_buffer:
            (loss, out) = encoder.loss(batch)
            sum_loss += loss.item()
        progress.set_description(f'Encoder loss {sum_loss}')

        if epoch % 10 == 0:
            save_image(out[:4], './encoder/vae.png')
            save_image(replay_buffer.items[:4], './encoder/truth.png')
            save_model(encoder.vae, encoder.optimizer, './encoder/model')

        loss_encoder.add_loss(sum_loss)
        loss_encoder.save()
    return encoder

def get_trained_encoder():
    encoder = Vae()
    load_model(encoder.vae, encoder.optimizer, './encoder/model')
    return encoder

