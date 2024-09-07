
class BestModel:
    def __init__(self) -> None:
        self.model = None
        self.score = None

    def set_model(self, model, score):
        if self.score is None or self.score < score:
            self.model = model
            self.score = score

    def get_model(self):
        return self.model
