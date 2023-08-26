import torch
from optimization_playground_shared.rl.Buffer import Buffer
from optimization_playground_shared.rl.atari_env import Env
from v1.agent import Agent
from v1.config import Config
import torch.nn as nn
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.plot.Plot import Plot, Image
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader
from optimization_playground_shared.rl.actor_critic import ActorCritic, ValueCritic, Episode, loss
import random
from typing import List
from torch.distributions import Categorical

metrics = Tracker("dreamer-v1-encoder")
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class TensorMapTracker:
    def __init__(self, size) -> None:
        self.tensors = [
            None for _i in range(size)
        ]

    def add(self, *args):
        for index, i in enumerate(args):
            i = i.unsqueeze(0)
            if self.tensors[index] is None:
                self.tensors[index] = i
            else:
                self.tensors[index] = torch.concat((
                    self.tensors[index],
                    i
                ), dim=0)


config = Config(
    z_size=128,
    image_size=80,
    action_size=9,
    device=device,
)
experience = Buffer()
agent = Agent(config)

"""
-> Tensors that then can be added to, but they keep the same indexes.
"""

tensor_batch = TensorMapTracker(
    5
)


def get_training_data():
    global agent, tensor_batch
    env = Env(
        config.image_size
    )
    entries = env.agent_play(agent).entries
    agent.reset()
    for i in entries:
        tensor_batch.add(
            i.state.to(device),
            i.next_state.to(device),
            torch.tensor([i.action], device=device),
            agent.last_state[0],  # batch 0
            torch.tensor([i.reward], device=device)
        )
        with torch.no_grad():
            agent.get_action(
                i.state.unsqueeze(0),
            )
    return tensor_batch


def train_encoder():
    tensor_batch = get_training_data()
    for epoch in range(1_000):
        sum_loss = 0
        dataset_size = 0
        if epoch % 10 == 0:
            # add more training data to the mix
            tensor_batch = get_training_data()
        for (state, next_state, action, encoded_state, reward) in get_dataloader(tensor_batch.tensors, batch_size=64):
            (predicted_state, predicted_reward) = agent.forward(state, encoded_state, action)
            recovered_next_state = agent.world_model.decode_latent(
                predicted_state
            )
            loss_reward = nn.L1Loss(reduction='sum')(
                predicted_reward,
                reward,
            )
            loss_construction = nn.L1Loss(reduction='sum')(
                recovered_next_state,
                next_state
            )

            agent.world_model.optimizer.zero_grad()
            loss = loss_reward + loss_construction
            loss.backward()
            agent.world_model.optimizer.step()
            dataset_size += recovered_next_state.shape[0]
            sum_loss += loss.item()

        idx = random.randint(0, next_state.shape[0] - 1)
        true_next_state = next_state[idx]
        recovered_next_state = recovered_next_state[idx]
        inference = Plot().plot_image([
            Image(
                image=recovered_next_state.detach().to(torch.device('cpu')).numpy(),
                title='recovered'
            ),
            Image(
                image=true_next_state.detach().to(torch.device('cpu')).numpy(),
                title='truth'
            ),
        ], f'inference.png')
        calculated_loss = sum_loss/dataset_size
        print(f"loss: {calculated_loss}".format(calculated_loss=calculated_loss))
        metric = Metrics(
            epoch=epoch,
            loss=calculated_loss,
            training_accuracy=None,
            prediction=Prediction.image_prediction(
                inference
            )
        )
        print((calculated_loss, metric.loss))
        assert calculated_loss == metric.loss,"error "+ type(calculated_loss)
#        print(metric.loss)
        metrics.log(metric)
        break


def train_behaviors():
    """
    Use actor critic with model predictions
    """
    actor_critic = ActorCritic(
        config.z_size,
        config.action_size
    )
    actor_critic.actor.to(config.device)
    actor_critic.critic.to(config.device)

    episode = Episode()
    predictions: List[ValueCritic] = []
    sum_reward = 0
    for (state, _, action, encoded_state, _) in get_dataloader(tensor_batch.tensors, batch_size=1):
        # what if we .... imagine!
        (predicted_state, predicted_reward) = agent.forward(state, encoded_state, action)
        actor = actor_critic.actor.forward(predicted_state)
        critic = torch.zeros((1), device=device)
        # planning into the future ....
        theta = 8
        for n in range(theta):
            critic += 0.99 ** (
                n - theta
            ) * predicted_reward[0] + (
                0.99 ** (theta - n) * actor_critic.critic(predicted_state)[0]
            )
            action = Categorical(
                actor_critic.actor.forward(predicted_state)
            ).sample()
            (predicted_state, predicted_reward) = agent.transition(
                predicted_state,  
                torch.tensor([[action]], device=device)
            )
        # sum it 
        predictions.append(ValueCritic(
            actor,
            critic
        ))
        episode.add(predicted_reward)
        sum_reward += predicted_reward

    calculated_loss = loss(actor_critic, predictions, episode, device=device)
    print(f"sum reward: {sum_reward.item()} loss: {calculated_loss}")


if __name__ == "__main__":
    actions = get_training_data()
    encoded_loss = train_encoder()
    behavior = train_behaviors()
    print(behavior)
