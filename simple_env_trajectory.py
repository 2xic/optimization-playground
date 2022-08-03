

from loss import Loss
from model import Model
from q_value_planning import Q_valuePlanning
from simple_env import SimpleEnv


def play(player: Model):
    env = SimpleEnv()
    loss = Loss()
    planner = Q_valuePlanning(player)

    iterations = 0

    while not env.done():
        action = planner.rollout(env.env)
        (state, reward, action, gamma) = env.step(action)
        loss.store_trajectory(
            state,
            action,
            reward,
            gamma
        )
        iterations += 1
    
    assert iterations > 0

    return (
        loss.iterate(player),
        loss,
    )
