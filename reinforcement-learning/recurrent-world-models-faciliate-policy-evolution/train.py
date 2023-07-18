from env import Env

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
        pass

    def encode(self, observartion):
        pass

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

if __name__ == "__main__":
    pass


#Env().save_observation()
