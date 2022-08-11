import numpy as np
from enum import Enum


"""
Should be a version of 
Rock-paper-scissors +
"""

class RockPaperScissor:
    def __init__(self) -> None:
        # rock, paper, scissor
        # optimal policy
        self.policy = [
            0.4, 0.4, 0.2
        ]
    
    def play(self, action):
        env_action = np.random.choice(3, 1, p=[0.1, 0.3, 0.6])[0]

        agent_action = Gesture(action)
        env_action = Gesture(env_action)

        return agent_action.winner(env_action)
        

class Gesture:
    def __init__(self, action) -> None:
        self.PAPER = "PAPER"
        self.ROCK = "ROCK"
        self.SCISSOR = "SCISSOR"
        self.mapping = {
            0:self.ROCK,
            1:self.PAPER,
            2:self.SCISSOR,
        }
        
        self.gesture = self.mapping[action] 

    def winner(self, gesture):
        """
            There is an error here
            In the paper they give double the reward / loss for scissor 
        """
        amplifier = 2 if (
            gesture.gesture == gesture.SCISSOR
            or self.gesture == gesture.SCISSOR
        ) else 1
        if (self.gesture == gesture.gesture):
            return 0 
        elif self.gesture == self.PAPER and gesture.gesture == gesture.ROCK:
            # we win 
            return 1 * amplifier
        elif self.gesture == self.SCISSOR and gesture.gesture == gesture.PAPER:
            # we win 
            return 1 * amplifier
        elif self.gesture == self.ROCK and gesture.gesture == gesture.SCISSOR:
            # we win 
            return 1 * amplifier
        else:
            return -1 * amplifier

class Actions(Enum):
    ROCK = 0
    PAPER = 1
    SCISSOR = 2
