
class Parameter:
    def __init__(self, name, from_value, to_value, steps=10, cast=None) -> None:
        assert from_value < to_value, f"{from_value} is not lower than {to_value} ({name})"
        self.name = name
        self.from_value = from_value
        self.to_value = to_value
        self.steps = steps
        self.step_size = (to_value - from_value) / steps
        # 
        self.value = self.from_value
        self._cast = cast


    def cast(self, value):
        if self._cast is None:
            return value
        else:
            # I don't wanna get stuck :(
            if self._cast == int:
                return round(value)            
            else:
                return self._cast(value)

    def get_values(self):
        values = []
        current = self.from_value
        while current < self.to_value:
            values.append(current)
            current = self._apply(current)
        return values

    def increment(self):
        if not self.done:
            self.value = self._apply(self.value)

    @property
    def done(self):
        return self.to_value <= self.value

    def _apply(self, current):
        output = self.cast(current + self.step_size)
        assert current < output, "Current is not less than the output"
        return output
