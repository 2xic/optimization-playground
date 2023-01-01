from .Parameter import Parameter

class PlotParameter:
    def __init__(self, parameter: Parameter, config={}) -> None:
        self.parameter = parameter
        self.config = config
