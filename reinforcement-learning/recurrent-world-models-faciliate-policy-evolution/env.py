import gymnasium as gym
from PIL import Image 

class Env:
    def __init__(self):
        self.env = gym.make("ALE/Breakout-v5")

    def reset(self):
        observation, _info = self.env.reset()
        return observation

    def observation(self, debug=False):
        observation, info = self.env.reset(seed=42)
        for _ in range(1000):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset()
        self.env.close()

    def save_observation(self, observation):
        data = Image.fromarray(observation)
        data.save('observation.png')
        
