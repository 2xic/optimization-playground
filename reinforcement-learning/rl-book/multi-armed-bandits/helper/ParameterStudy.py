

class ParameterStudy:
    def __init__(self, name, values, apply_parameter, apply_update) -> None:
        self.x = []
        self.y = []

        self.name = name
        self.apply_parameter = apply_parameter
        self.apply_update = apply_update
        self.values = values
        self.index = 0

    def add_results(self, value):
        self.x.append(self.values[self.index])
        self.y.append(value)

    def is_done(self):
        return self.index == len(self.values)

    def next(self):
        self.index += 1

    def get_agent(self) -> None:
        active_agent = self.apply_parameter(self.values[self.index])

        return active_agent
