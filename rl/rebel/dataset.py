

from pbs import PBS


class Dataset:
    def __init__(self) -> None:
        self.value = []
        self.policy = []

    def add_value_net(self, pbs: PBS, value):
        self.value.append([pbs, value])

    def add_policy_net(self, pbs: PBS, value):
        self.policy.append([pbs, value])


