

class MaxItem:
    def __init__(self) -> None:
        self.numeric_value = float('-inf')
        self.value = None

    def max(self, numeric, value):
        if (self.numeric_value < numeric):
            self.numeric_value = numeric
            self.value = value
