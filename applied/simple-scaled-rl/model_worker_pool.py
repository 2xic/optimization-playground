"""
I want one model for each worker.

Somewhat inspired from https://openai.com/research/requests-for-research-2 where 
they want us to do parameter averaging.

1. Each model learns indecently.

2. We at some interval average the parameters back to a root model

3. Based on the average parameters this could be applied to each of the models 
again.

4. 
"""
import torch.optim as optim
import torch

class ModelPool:
    def __init__(self, n, contract_sub_model) -> None:
        self.n = n
        self.models = [
            contract_sub_model()
            for _ in range(n)
        ]
        self.optimizers = [
            optim.Adam(model.parameters())
            for model in self.models
        ]
        self.step_counter = 0
        self.rates = [
            0.5,
            0.25,
            0.1,
        ]
        self.epochs = 0

    def __getitem__(self, idx):
        return self.models[idx]

    def step_model(self):
        # assumes the models already have gradients backward passed
        for (model, optimizer) in zip(self.models, self.optimizers):
            optimizer.step()
            model.zero_grad()
        self.step_counter += 1

        if self.step_counter % 5 == 0:
            pass
        self.get_average_parameters()

    def get_average_parameters(self):
        # update rates 
        rate_index = min(self.epochs // 1_00, len(self.rates) - 1)
        rate = self.rates[rate_index]

        # iterate over all parameters
        for i in zip(*list(map(lambda x: x.parameters(), self.models))):
            base_params = torch.zeros_like(i[0])
            for sub_model in i:
                base_params += sub_model.data
           # print(base_params)
            avg = base_params / self.n
            # ^ adjust based on this avg.
            for sub_model in i:
                sub_model.data.copy_(
                    (1 - rate) * sub_model.data +
                    rate * avg
                )
        print("Updated avg model params :D")
