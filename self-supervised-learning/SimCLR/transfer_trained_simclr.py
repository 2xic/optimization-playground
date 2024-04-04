from model import SimClrTorch, Net
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch
from torch import nn
import torch.nn as nn
from optimization_playground_shared.utils.CopyModelWeights import CopyModelWeights
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader

class TransferLearning(nn.Module):
    def __init__(self, backend, output_size=100, disable_params=True):
        super().__init__()
        self.disable_params = disable_params
        self.set_backend(backend)

        # Simple classifier to be similar to the baseline model
        self.classifier = nn.Sequential(
            nn.Linear(100, output_size),
            nn.LogSoftmax(dim=1),
        )

    def set_backend(self, backend):
        if self.disable_params:
            self.backend = backend.eval()
            # The conv layer should be freezed
            for params in list(self.backend.modules()):
                if isinstance(params, nn.Conv2d) or isinstance(params, nn.MaxPool2d):
                    params.requires_grad = False
        else:
            self.backend = backend

    def forward(self, x):
        x = self.backend(x)
        x = self.classifier(x)
        return x


reference_model = TransferLearning(Net(), output_size=10, disable_params=False)
reference_optimizer = torch.optim.Adam(reference_model.parameters())
reference_loop = TrainingLoop(reference_model, reference_optimizer)

transfer_model = TransferLearning(Net(), output_size=10, disable_params=False)
optimizer = torch.optim.Adam(transfer_model.parameters())
loop = TrainingLoop(transfer_model, optimizer)

copy_weights_data = CopyModelWeights()

def test_model(model: SimClrTorch, device):
    transfer_model.to(device)
    reference_model.to(device)

    copy_weights_data.update(
        transfer_model.backend,
        model.model
    )
    loop = TrainingLoop(transfer_model, optimizer)

    train_loader, test_loader = get_dataloader()

    """
    Transferred learned model
    """
    epoch_reference_loss = []
    epoch_simclr_loss = []
    for _ in range(10):
        (loss, _) = loop.train(train_loader)
        (reference_loss, _) = reference_loop.train(train_loader)

        epoch_simclr_loss.append(loss)
        epoch_reference_loss.append(reference_loss)
        copy_weights_data.epochs += 1
    # hacky to correct the weights updates
    copy_weights_data.epochs -= 1

    accuracy = loop.eval(test_loader)
    reference_accuracy = reference_loop.eval(test_loader)
    return accuracy.cpu().item(), reference_accuracy.cpu().item(), epoch_simclr_loss, epoch_reference_loss
