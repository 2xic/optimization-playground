
class PBS:
    """
    PBSs are defined by a common-knowledge belief distribution over states, 
    determined by the public observations shared by all agents and the policies of all agents.

    ^ quote form the paper
    """
    def __init__(self) -> None:
        pass

    def info_state(self):
        # the sequence of actions + observation in the env
        return []

    def actions(self):
        return []

    def is_leaf(self):
        return True

    def getRandomHistory(self):
        return PBS()

    def transition(self, policy, action):
        return PBS()

class InfoState:
    def __init__(self, state) -> None:
        self.value = 0
        self.state = state
