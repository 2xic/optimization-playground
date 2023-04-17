from .q_learning import Q_learning
from .sarsa import Sarsa
from helpers.play_cliff_walking_vs_agent import play_cliff_walking

if __name__ == "__main__":
    play_cliff_walking(
        Q_learning,
        Sarsa
    )
