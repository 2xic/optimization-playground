

"""
Algorithm 2
"""

class Rebel:
    def __init__(self, model) -> None:
        self.model = model

    def linear(self, pbs):
        while not self.is_terminal():
            G = self.construct_subgame()
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
                policy_t = None # update policy
                policy = (t / (t + 2)) * policy + (2 / (t + 2)) * policy_t
                G = self.set_leaf_values(
                    pbs,
                    policy_t,
                    self.model.value
                ) 
            
            # 


    def construct_subgame(self):
        """
        A subgame is defined by a root history h in a perfect-information game and all histories that can
        be reached going forward.
        - quote from paper
        """

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



    