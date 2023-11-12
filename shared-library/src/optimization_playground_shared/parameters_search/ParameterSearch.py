from .Parameter import Parameter

class ParameterSearch:
    def __init__(self, parameters: Parameter) -> None:
        self._parameters = parameters
        self.index = 0

    def parameters(self):
        value = {}
        for i in self._parameters:
            value[i.name] = i.value
        return value

    def step(self):
        """
        We increment one parameter at the time
        """
        self._parameters[self.index].increment()
        self.index = (self.index + 1) % len(self._parameters)

    @property
    def done(self):
        return self._parameters[-1].done 
