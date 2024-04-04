from Model import FixMatchModel, Net
from Dataloader import Cifar10Dataloader, RawCifar10Dataloader
from torch.utils.data import DataLoader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.plot.Plot import Plot, Figure
from Dataloader import Cifar10Dataloader, RawCifar10Dataloader
from torch.utils.data import DataLoader
import torch

dataset = Cifar10Dataloader()

train_loader = DataLoader(dataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=2)
raw_train_loader = DataLoader(RawCifar10Dataloader(dataset.dataset),
                          batch_size=256,
                          shuffle=True,
                          num_workers=2)

reference_model = Net()

fixmatch_model = FixMatchModel(Net())
fixmatch_model_optimizer = torch.optim.Adam(fixmatch_model.model.parameters())
_, test_loader = get_dataloader()

reference_optimizer = torch.optim.Adam(reference_model.parameters())
reference_loop = TrainingLoop(reference_model, reference_optimizer, loss=torch.nn.CrossEntropyLoss())

training_loss_mixmatch = []
training_loss_reference = []

training_accuracy_mixmatch = []
training_accuracy_reference = []

test_accuracy_mixmatch = []
test_accuracy_reference = []

for epoch in range(1_00):
    fixxmatch_accuracy = 0
    samples = 0
    loss_training = 0
    for (x, y, z) in train_loader:
        fixmatch_model_optimizer.zero_grad()
        (loss, accuracy) = fixmatch_model.training_step((x, y,z))
        loss.backward()
        fixmatch_model_optimizer.step()
        loss_training += loss.item()
        # accuracy
        fixxmatch_accuracy += accuracy
        samples += 1
        assert fixxmatch_accuracy <= samples, f"{fixxmatch_accuracy} vs {samples}"

    print(f"Epoch {epoch}, acc {fixxmatch_accuracy} / {samples}")

    training_loss, training_accuracy = reference_loop.train(raw_train_loader)
    test_accuracy = reference_loop.eval(test_loader)
    training_accuracy_mixmatch.append(
        (fixxmatch_accuracy / samples) * 100
    )
    training_accuracy_reference.append(
        training_accuracy
    )
    test_accuracy_reference.append(
        test_accuracy
    )
    training_loss_reference.append(
        training_loss
    )
    training_loss_mixmatch.append(
        loss_training
    )
    test_accuracy_mixmatch.append(
        TrainingLoop(fixmatch_model.model, None, None).eval(test_loader)
    )

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Training loss with MixMatch": training_loss_mixmatch,
                    "Training loss without MixMatch": training_loss_reference,
                },
                title="Loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Training accuracy with MixMatch": training_accuracy_mixmatch,
                    "Training accuracy without MixMatch": training_accuracy_reference,
                },
                title="Training accuracy",
                x_axes_text="Epoch",
                y_axes_text="Accuracy",
            ),
            Figure(
                plots={
                    "Test accuracy with MixMatch": test_accuracy_mixmatch,
                    "Test accuracy without MixMatch": test_accuracy_reference,
                },
                title="Test accuracy",
                x_axes_text="Epoch",
                y_axes_text="Accuracy",
            )
        ],
        name='results.png'
    )

