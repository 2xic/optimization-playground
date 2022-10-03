import random
from tracemalloc import take_snapshot
from dataset import Dataset
import numpy as np
from fake_policy import FakePolicy
from pbs import PBS

"""
Algorithm 2
"""

class Rebel:
    def __init__(self, policy, value) -> None:
        self.policy = policy
        self.value = value
        self.dataset = Dataset()
        self.count = 0
    """
    So the way I understand this is the following.
    - PBS is basically a state distribution encoded by a model with the current state.
        - The current state is used to know legal actions.
    - In the paper the input is also a value and policy parameters.
    - In addition to training data 
        - Which is used for the re-training.
    """
    def linear(self, pbs: PBS):
        """
        check if the pbs is a terminal node
            - should just check if you can do any more legal actions, or timeout
        """
        while not self.is_terminal():
            # "do a rollout" from the current pbs
            G = self.construct_subgame(pbs)
            
            # initialize a policy given the sub game, and policy
            # *paper sets pi, pi^t_warm here
            policy, policy_t = FakePolicy()(None), FakePolicy()(None)

            # Set the leaf nodes from the nn
            G = self.set_leaf_values(
                pbs,
                self.policy,
                self.value
            )
            value = self.compute_evg(G, policy_t) # compute ev based on the policy and current sub game

            T = 10 # where is this set ?
            t_sample = random.randint(1, T - 1) # linear sampling to T

            visited_pbs = []
            random_pbs = None
            for t in range(1, T):
                #print(t, t_sample)
                if t == t_sample:
                    """
                    sample a leaf from the sub game
                    """
                    random_pbs = self.sample_leaf(G, policy)

                
                # Update policy 
                # update policy -> This should use cfr (I think ?)
                #                   ^ or is this the model, and EV the CFR ? 
                #                   No in the paper they write "On each iteration t, CFR-D determines a policy profile πin the subgame."
                policy_t = FakePolicy()(pbs) 

                policy = (t / (t + 2)) * policy + (2 / (t + 2)) * policy_t

                # outputs new policy
                G = self.set_leaf_values(
                    pbs,
                    policy_t,
                    self.value
                )

                value = (t / (t+2)) *  value + (2 / (t + 2)) * self.compute_evg(G, policy_t)
                visited_pbs.append(pbs)

            # Add to value net training data
            self.dataset.add_value_net(pbs, value)

            for i in visited_pbs:
                self.dataset.add_policy_net(i, FakePolicy()(i))

            if random_pbs is None:
                raise Exception("This should not happened")

            pbs = random_pbs

            self.count += 1
        self.count = 0

    def  is_terminal(self):
        return not (self.count < 10)

    def compute_evg(self, game, policy):
        # TODO: should ofc not be static
        return 10

    def construct_subgame(self, pbs):
        """
        A subgame is defined by a root history h in a perfect-information game and all histories that can
        be reached going forward.
        - quote from paper

        Basically do a rollout
        """
        return pbs

    def set_leaf_values(self, pbs: PBS, policy, value):
        """
            pass
        """
        if pbs.is_leaf():
            for i in pbs.info_state():
                # set the value of the info state to be value(s_i | pbs, value_policy)
                # <- need to look at notion in section 3
                # look at section 5.1 also.
                #pass
                i.state_value = self.value(i.state) # v(s_i | pbs, policy)
        else:
            for action in pbs.actions():
                self.set_leaf_values(
                    # transition is learned (I think ?).
                    pbs.transition( 
                        # pbs = self
                        policy, 
                        action
                    ),
                    policy,
                    value
                )
        return pbs

    def sample_leaf(self, subGame, policy, epsilon=.025):
        search_index = random.randint(0, 1)
        h = subGame.getRandomHistory()

        while not h.is_leaf():
            c = np.rand.random()

            for i in range(len(h)):
                if i == search_index and c < epsilon:
                    # sample an action uniformly
                    pass
                else:
                    # sample an action according to policy
                    pass
            h = transition(h, a)
        return h

    