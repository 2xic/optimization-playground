
class CopyModelWeights:
    def __init__(self):
        self.rates = [
            0.5,
            0.25,
            0.1,
        ]
        self.epochs = 0

    def update(self, base_model, update_model):
        rate_index = min(self.epochs // 1_00, len(self.rates) - 1)
        rate = self.rates[rate_index]
        for (base_model, update_model_param) in zip(base_model.parameters(), update_model.parameters()):
            base_model.data.copy_(
                (1 - rate) * base_model.data +
                (rate) * update_model_param.data
            )
        self.epochs += 1
