"""
- We need quite a few model components
    - One world model framework
    - One actor critic framework
- Replay with env environment data

Training will be
1. Train the image encoder
2. Train the reward model (?)
^ supervised- nothing fancy


"""
from dataclasses import dataclass
import torch
from optimization_playground_shared.rl.atari_env import Env
from train_world_model import WorldModel, RecurrentModel
from train_actor_critic import ActorCriticModel


"""
Checklist
[X] Recurrent model
[X] Representation model
[X} Transition model
[X] Image predictor
[X] Reward predictor
[X] Dicsount predictor
"""

@dataclass
class EnvStateActionPairs:
    state: torch.Tensor
    next_state: torch.Tensor
    action: int
    reward: float

def get_replay():
    env = Env()
    replay = []


    for (next_observation, observation, action, reward) in env.random_play():
        replay.append(
            EnvStateActionPairs(
                state=observation,
                next_state=next_observation,
                action=action,
                reward=reward
            )
        )
    return replay

def train():
    action_size = 9
    latent_size = 128

    replay = get_replay()
    model = WorldModel()
    recurrent_model = RecurrentModel(
        latent_size=latent_size,
        action_size=action_size
    )
    actor_critic = ActorCriticModel(
        latent_size=latent_size,
        hidden_size=64
    )

    # train world model
    hidden_state = recurrent_model.initial_state(
        batch_size=1
    )
    for i in replay:
        latent = model.representation(i.state.unsqueeze(1), hidden_state)
        actions = torch.zeros((1, action_size))
        actions[0][i.action] = 1

        (_, hidden_state) = recurrent_model.forward(
            latent,
            actions,
            hidden_state
        )
        recovered_image = model.image_predictor(hidden_state, latent)
        (reward_predictor, discounter) = actor_critic.forward(latent, hidden_state)
        transition_output = model.transition(latent)
        """
        Improved loss functions needed.

        We also need a rollout function
        """


if __name__ == "__main__":
    train()
