import numpy as np

class MixUp:
    def __init__(self) -> None:
        self.alpha = 0.2

    """
    TODO: make this more tensor friendly
    """
    def __call__(self, x, y, x2, y2):
        batch_size = 1
        l = np.random.beta(self.alpha, self.alpha, batch_size)
        print(l)
        l = np.maximum(l, np.ones(l.shape) - l)

        new_x = x * l + (1 - l) * x2
        new_y = y * l + (1 - l) * y2

        return (
            new_x,
            new_y
        )
