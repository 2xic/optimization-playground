from abc import ABC, abstractmethod
 
class Model(ABC):

    @abstractmethod
    def core(self, _state, _action):
        pass

    @abstractmethod
    def transition(self, _state, _action):
        pass

    @abstractmethod
    def outcome(self, _state, _action):
        pass

    @abstractmethod
    def value(self, _state):
        pass
    
    @abstractmethod
    def encode(self, _state):
        pass
