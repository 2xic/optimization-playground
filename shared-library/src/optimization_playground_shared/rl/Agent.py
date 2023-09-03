import abc

class Agent(metaclass=abc.ABCMeta):
    def ___init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def get_action(self, state):
        pass
