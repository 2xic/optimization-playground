

class BestModelParameters:
    def __init__(self) -> None:
        self.state = None
        self.loss = float('inf')

    def set_loss_param(self, loss, state):
        if loss < self.loss:
            loss = self.loss
            self.state = state
