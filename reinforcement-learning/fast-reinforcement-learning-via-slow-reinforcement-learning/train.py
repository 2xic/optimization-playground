from optimization_utils.envs.TicTacToe import TicTacToe
import torch
from optimization_playground_shared.rl.epsilon import Epsilon
from model import Model
from collections import namedtuple
import torch.optim as optim
import random


EPOCHS = 1_000
ROLLOUTS = 10

def get_env_state(env, action=-1, reward=-1, d=-1):
    return torch.concat((
        env,
        torch.tensor([[reward, action, d]])
    ), dim=1)
    
def get_legal_actions(model, state, env):
    model_output, hidden_state = model(state)
    for i in range(0, env.action_space):
        if i not in env.legal_actions:
            model_output[0][i] = 0
    action = torch.argmax(model_output[0]).item()
#    print(model_output, action)
#    exit(0)
    return action, hidden_state


def play_random_agent():
    env = TicTacToe(n=3, is_auto_mode=True)
    policy = Epsilon(
        decay=1
    )
    reward_batch = []
    for _ in range(EPOCHS):
        sum_reward = 0
        for _ in range(ROLLOUTS):
            env.reset()
            while not env.done and env.winner is None:
                action = policy.action(
                    env.legal_actions
                )
                env.step(action)
            sum_reward += env.winner or 0
        reward_batch.append(
            sum_reward
        )
    return reward_batch

def train(iteration):
    env = TicTacToe(n=3, is_auto_mode=True)
    policy = Epsilon(
        decay=0.995
    )

    model = Model(env.action_space)
    optimizer = optim.Adam(model.parameters())

    reward_batch = []
    for epoch in range(EPOCHS):
        reward = 0
        action = 0
        hidden_state = None

        entry = namedtuple('entry', 'state hidden_state action reward')
        batch = []

        sum_reward = 0
        for _ in range(ROLLOUTS):
            env.reset()
            local_batch = []
            while not env.done and env.winner is None:
                state = get_env_state(env.env, reward=reward, action=action)
                action = None
                while action not in env.legal_actions:
                    random_action = policy.action(
                        env.legal_actions
                    )
                    model_action = None
                    prev_hidden_state = hidden_state.detach() if hidden_state is not None else None
                    if random_action is None:
                        model_action, hidden_state = get_legal_actions(model, state, env)  #"model(state)
                    action = list(filter(lambda x: x is not None, [random_action, model_action]))[0]
                
                (
                        _,
                        reward,
                        action,
                        _
                ) = env.step(action)
                local_batch.append(
                    entry(
                        state=state,
                        action=action,
                        hidden_state=prev_hidden_state,
                        reward=reward
                    )
                )
            #for index, i in enumerate(local_batch):
            #    i._replace(reward=(env.winner or 0) * (1+ index) / len(local_batch))
            batch += local_batch
            sum_reward += env.winner or 0
           # print(env.winner, env.winner or 0, sum_reward)
        
        """
        I guess in theory, we could output an reward matrix, and based on that calculate the error without this loop.
        """
        optimizer.zero_grad()
        loss = 0
        batch = random.sample(batch, min(len(batch), 64))
        for i in batch:
            predicted, _ = model(i.state, hidden=i.hidden_state)
            wanted = None
            with torch.no_grad():
                wanted, _ = model(i.state, hidden=i.hidden_state)
                wanted[0][i.action] = i.reward
            loss += torch.nn.MSELoss()(predicted, wanted)
        loss_normalized = loss #/ len(batch)
        #print(loss)
        loss_normalized.backward()
        optimizer.step()

        if epoch % 32 == 0:
            print(f"({iteration}) loss = {loss_normalized.item()}, {sum_reward}, eps={policy.epsilon}")
        reward_batch.append(sum_reward)
    return reward_batch
