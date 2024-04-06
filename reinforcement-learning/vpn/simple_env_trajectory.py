from epsilon import Epsilon
from loss import Loss
from model import Model
from q_value_planning import Q_valuePlanning
from optimization_utils.envs.SimpleEnv import SimpleEnv

def play(player: Model, epsilon: Epsilon):
    env = SimpleEnv()
    loss = Loss()
    planner = Q_valuePlanning(player)

    iterations = 0
    total_reward = 0

    while not env.done():
        epsilon_action = epsilon.action()
        action = planner.rollout(env.env) if epsilon_action is None else epsilon_action
      #  print((epsilon_action, action))

        (state, reward, action, gamma) = env.step(action)
        loss.store_trajectory(
            state,
            action,
            reward,
            gamma
        )
        iterations += 1
        total_reward += reward
    
    assert iterations > 0

    return (
        loss.iterate(player),
        loss,
        total_reward
    )

