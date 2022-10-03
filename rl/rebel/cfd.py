# rewrite of an cfd I wrote many years ago..
# https://github.com/2xic-archive/Notebooks/blob/master/counterfactual_regret_minimization.ipynb

import numpy as np

class CFD:
    def __init__(self) -> None:
        # TODO: Should probably make it generic.
        self.ROCK = 0
        self.PAPER = 1
        self.SCISSORS = 2
        self.NUM_ACTIONS = 3    
    
        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        self.strategy = np.zeros(self.NUM_ACTIONS)
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)

    def get_strategy(self):
        normalizing_sum = 0
        for action in range(self.NUM_ACTIONS):
            self.strategy[action] = self.regret_sum[action] if(self.regret_sum[action] > 0) else 0
            normalizing_sum += self.strategy[action]
        for action in range(self.NUM_ACTIONS):
            if(normalizing_sum > 0):
                self.strategy[action] /= normalizing_sum
            else:
                self.strategy[action] = 1 / self.NUM_ACTIONS
            self.strategy_sum[action] += self.strategy[action]
        return self.strategy

    def get_action(self, strategy):
        r = np.random.rand()
        a = 0
        cumulative_probability = 0
        while a < self.NUM_ACTIONS - 1:
            cumulative_probability += strategy[a]
            if(r < cumulative_probability):
                break
            a += 1
        return a

    def train(self, target, epochs=1000):
        action_utility = np.zeros(self.NUM_ACTIONS)
        output = [
            [],
            [],
            []
        ]
        for _ in range(epochs): 
            # Get regret-matched mixed-strategy actions
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            optimal_action = self.get_action(target)
        
            # Compute action utilities
            action_utility[optimal_action] = 0        
            

            # TODO: I don't remember this part.
            state = (0 if(optimal_action == self.NUM_ACTIONS - 1) else optimal_action + 1)
            state1 = (self.NUM_ACTIONS - 1 if(optimal_action == 0) else optimal_action - 1)
            
            action_utility[state] = 1
            action_utility[state1] = - 1
            
            # Accumulate action regrets
            for action in range(self.NUM_ACTIONS):
                self.regret_sum[action] += action_utility[action] - action_utility[my_action]
                output[action].append(self.regret_sum[action])
        
        return self.get_average_strategy()

    def get_average_strategy(self):
        average_strategy = np.zeros(self.NUM_ACTIONS)
        normalizing_sum = 0
        for action in range(self.NUM_ACTIONS):
            normalizing_sum += self.strategy_sum[action]
        for action in range(self.NUM_ACTIONS):
            if(normalizing_sum > 0):
                average_strategy[action] = self.strategy_sum[action] / normalizing_sum
            else:
                average_strategy[action] = 1.0 / self.NUM_ACTIONS
        return average_strategy
