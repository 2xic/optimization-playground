import torch
import copy
from model import Learning2Learn


class TorchLearning2LearnOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, use_trained=True):
        defaults = dict(lr=lr)
        super(TorchLearning2LearnOptimizer, self).__init__(params, defaults)
        self.model = Learning2Learn(n_features=1,
                                    hidden_size=64,
                                    num_layers=2,
                                    dropout=0.2)
        if use_trained:
            self.model.load_from_checkpoint("checkpoints_l2l/lightning_logs/version_0/checkpoints/epoch=999-step=10000.ckpt",
                                            n_features=1,
                                            hidden_size=64,
                                            num_layers=2,
                                            dropout=0.2)
        self.epoch = 0

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for index, p in enumerate(group['params']):
                if p.grad is not None:
                    pass
                # we have gradient
                d_p = p.grad.data
                # we want to minimize the gradient.
                # so we feed that into the neural network
                # then update the weights.
                update_g = (self.model(lambda x: (x * d_p).sum())).item()

           #     print(update_g)

                #print(d_p)
                d_p = d_p.add(update_g) #0.01)
                #print(d_p)
                #print(update_g)
                #print(group['lr'])
                #exit(0)

                p.data.add_(-group['lr'], d_p)
                #p.data.add_(d_p)

                if index == 0 and self.epoch % 10 == 0:
                    print(p.data, update_g, d_p, self.epoch)
        self.epoch += 1
    
        return loss
