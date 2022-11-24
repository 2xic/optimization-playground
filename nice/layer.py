

class AddictiveCouplingLayer:
    def __init__(self) -> None:
        self.model = lambda x: x

    def forward(self, x_1, x_2, prev=None):
        y_1 = x_1 if prev is None else prev
        y_2 = x_2 + self.model(x_1)

        return (
            y_1,
            y_2
        )

    def backward(self, y_1, y_2):
        x_1 = y_1
        x_2 = y_2 - self.model(y_1)

        return (
            x_1,
            x_2
        )
