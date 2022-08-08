

"""
Algorithm 2
"""

class Rebel:
    def __init__(self, model) -> None:
        self.model = model

    """
    So the way I understand this is the following.
    - PBS is basically a state distribution encoded by a model with the current state.
        - The current state is used to know legal actions.
    - In the paper the input is also a value and policy parameters.
    - In addition to training data 
        - Which is used for the re-training.
    """
    def linear(self, pbs):
        """
        check if the pbs is a terminal node
            - should just check if you can do any more legal actions, or timeout
        """
        while not self.is_terminal():
            # "do a rollout" from the current pbs
            G = self.construct_subgame(pbs)
            # initialize a policy given the sub game, and policy
            # *paper sets pi, pi^t_warm here

            # Set the leaf nodes from the nn
            G = self.set_leaf_values(
                pbs,
                self.model.policy,
                self.model.value
            )

            policy, policy_t = None, None

            # compute ev 
            T = 10 # where is this set ?
            t_sample = 1 # linear sampling to T

            for t in range(1, T):
                if t == t_sample:
                    # sample the leaf
                    pass
                    
                policy_t = None # update policy -> This should use cfr
                policy = (t / (t + 2)) * policy + (2 / (t + 2)) * policy_t
                # outputs new policy
                G = self.set_leaf_values(
                    pbs,
                    policy_t,
                    self.model.value
                ) 
            
            # 


    def construct_subgame(self, pbs):
        """
        A subgame is defined by a root history h in a perfect-information game and all histories that can
        be reached going forward.
        - quote from paper
        """
        pass

    def set_leaf_values(self, pbs, policy, value):
        """
            pass
        """
        if pbs.is_leaf():
            for i in pbs.info_state():
                # set the value of the info state to be value(s_i | pbs, value_policy)
                # <- need to look at notion in section 3
                # look at section 5.1 also.
                pass
        else:
            for i in pbs.actions():
                self.set_leaf_values(
                    # transition is learned.

                    pbs.transition(policy, i),
                    policy,
                    value
                )
        
    def is_terminal(self):
        return False



    