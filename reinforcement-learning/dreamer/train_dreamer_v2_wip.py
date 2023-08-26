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
from v2.train_world_model import WorldModel, RecurrentModel
from v2.train_actor_critic import RewardModel, ActorCriticModel
from shared.utils import RewardPrinter
import torch.optim as optim
import torch
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics

torch.set_default_device('cuda:0')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics = Tracker("dreamer")
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


action_size = 9
latent_size = 128
model = WorldModel()
recurrent_model = RecurrentModel(
    latent_size=latent_size,
    action_size=action_size
)
reward_model = RewardModel(
    latent_size=latent_size,
    hidden_size=64
)
actor_critic_model = ActorCriticModel(
    latent_size=latent_size,
    hidden_size=64,
    actions=action_size
)


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


def model_replay():
    with torch.no_grad():
        env = Env()
        
        def model_forward(state, hidden_state):
            sate_encoded = model.representation(state.unsqueeze(1), hidden_state)
            next_actions = actor_critic_model.get_actions(
                sate_encoded,
                hidden_state
            )
            (_, hidden_state) = recurrent_model.forward(
                sate_encoded,
                next_actions,
                hidden_state
            )
            return torch.argmax(next_actions, dim=1)[0], (hidden_state, )
        sum_reward, state_actions = env.iter_play(model_forward, recurrent_model.initial_state(1))
        replay = []
        for (next_observation, observation, action, reward) in state_actions:
            replay.append(
                EnvStateActionPairs(
                    state=observation,
                    next_state=next_observation,
                    action=action,
                    reward=reward
                )
            )
        return sum_reward, replay

class Loss:
    def __init__(self) -> None:
        pass

    def loss_representation_model(self, predicted_image, true_image):
        assert predicted_image.shape == true_image.shape, f"{predicted_image.shape} vs {true_image.shape}"
        # todo should be Kl loss
        return torch.nn.MSELoss()(predicted_image, true_image.to(device))

    def loss_dynamics_model(self, predicted_z, true_z):
        # todo should be Kl loss
        return torch.nn.MSELoss()(predicted_z, true_z)

    def loss_transition_model(self):
        pass

    def loss_reward_model(self):
        pass

    def loss_value_function(self, predicted, truth):
        return torch.nn.MSELoss()(predicted, truth.to(device))


def train():

    loss = Loss()
    world_optimizer = optim.Adam(model.parameters())

    # train world model
    hidden_state = recurrent_model.initial_state(
        batch_size=1
    )
    actor_optimizer = optim.Adam(
        actor_critic_model.action_distribution.parameters())
    critic_optimizer = optim.Adam(actor_critic_model.critic.parameters())

    reward_printer = RewardPrinter()
    for epoch in range(100):
        batch_world_model_loss = 0
        batch_actor_critic_loss = 0
        sum_reward, replay = model_replay()
        for index, i in enumerate(replay[:-1]):
            latent = model.representation(i.state.unsqueeze(1), hidden_state)
            actions = torch.zeros((1, action_size))
            actions[0][i.action] = 1
            previous_hidden_state = hidden_state.clone()
            """ 
            TODO: Considering putting the recurrent component as part of the learning. input.
            """
            (_, hidden_state) = recurrent_model.forward(
                latent,
                actions,
                hidden_state
            )
            recovered_image = model.decode_latent(latent, hidden_state)
            dynamic_predictor = model.dynamic_predictor(hidden_state, latent)
            """
            Improved loss functions needed.

            We also need a rollout function
            """

            # Latent (encoded state) -> Decoded back into state
            representation_loss = loss.loss_representation_model(
                recovered_image,
                i.state.unsqueeze(1)
            )

            # dynamic predictor -> predict the next representation given the previous one
            # the loss in the paper is a bit unclear as both have index `t`, but quite sure they meant t + 1
            """
            Oh, actually I see what they did. They try to make sure the model are aligned.

            So it should actually be KL div between dyn and rep and vice versa.
            """
            next_predictor = None
            with torch.no_grad():
                next_predictor = model.representation(
                    replay[index + 1].state.unsqueeze(1), hidden_state)
            dynamic_loss = loss.loss_dynamics_model(
                dynamic_predictor, next_predictor)
            sum_loss = representation_loss + dynamic_loss

            batch_world_model_loss += sum_loss.item()

            world_optimizer.zero_grad()
            sum_loss.backward()
            world_optimizer.step()
            # remove the hidden_state from the learning -> Is this correct ? Maybe
            hidden_state = hidden_state.detach()

            """
            Actor critic learning
            """
            # imagination what is possible
            # -> get out action distribution (input to other models)
            # ->
            # I imagine 4 steps into the future
            imagination_hidden_state = previous_hidden_state.detach()
            imagined_state = model.representation(
                replay[index + 1].state.unsqueeze(1), imagination_hidden_state)
            value = 0
            for i in range(index, index + 4):
                (reward_predictor, discounter) = reward_model.forward(
                    imagined_state,
                    imagination_hidden_state
                )
                next_actions = actor_critic_model.get_actions(
                    imagined_state, imagination_hidden_state
                )
                # action = next_actions.argmax(dim=1)
                (_, imagination_hidden_state) = recurrent_model.forward(
                    imagined_state,
                    next_actions,
                    imagination_hidden_state
                )
                imagined_state = model.dynamic_predictor(
                    imagination_hidden_state, imagined_state)

                value += (
                    # 0.99 = gamma
                    0.99 ** (i - index)
                    * reward_predictor +
                    # imagination
                    (0.99 ** (index + 4 - index)) * \
                    actor_critic_model.get_critic_value(imagined_state)
                )
            """
            Actor should predict good actions
            Critic should know the value of a state
            """
            # the action loss ?
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss_value = loss.loss_value_function(value, actor_critic_model.get_critic_value(
                imagined_state
            ))
            batch_actor_critic_loss += loss_value.item()
            loss_value.backward()
            actor_optimizer.step()
            critic_optimizer.step()
        reward_printer.add_reward(
            sum_reward
        )
        print(f"world model loss : {batch_world_model_loss}, actor critic loss {batch_actor_critic_loss}")
        print(reward_printer.print())
        metrics.log(Metrics(
            epoch=epoch,
            loss=(batch_world_model_loss + batch_actor_critic_loss),
            training_accuracy=sum_reward
        ))

if __name__ == "__main__":
#    model_replay()
    train()
