from train import Agent
from train_encoder import get_trained_encoder
from train_predictor import get_trained_predictor
from optimization_playground_shared.rl.atari_env import Env
from utils import save_image

def eval_loop():
    """
    Play the env
    """
    env = Env()
    agent = Agent(
        get_trained_predictor(env),
        get_trained_encoder()
    )
    agent.explorer.epsilon = 0.1
    for index, observation in enumerate(env.eval_agent_play(agent)):
        print(index)
        save_image(observation, f'./eval/{index}.png')

if __name__ == "__main__":
    eval_loop()
