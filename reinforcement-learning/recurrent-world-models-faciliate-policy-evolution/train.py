from env import Env
from optimization_playground_shared.models.SimpleVaeModel import SimpleVaeModel
import torch.optim as optim
from PIL import Image
import numpy as np 
from torchvision.utils import save_image

class Rnn:
    def __init__(self) -> None:
        pass

    def initial_state(self):
        pass

    def forward(self, tensor):
        pass

    def action(self, tensor):
        pass

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
    
    def loss(self, observation):
        self.vae.zero_grad()

        out = self.vae.encode(observation)
        out = self.vae.decode(out)[:, :3, :observation.shape[2], :observation.shape[3]]
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
            z = self.vae.encode(observation) # vae encoding
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

def train_encoder():
    env = Env()
    encoder = Vae()
    for _ in range(10_000):
        observation = env.reset().unsqueeze(0)
        (loss, out) = encoder.loss(observation)
        print(
            loss
        )
#        Image.fromarray(out[0].detach().numpy().astype(np.int8)).save('test.png')
        save_image(out, 'test.png')



if __name__ == "__main__":
    train_encoder()


#Env().save_observation()
