from Model import FixMatchModel
from Dataloader import Cifar10Dataloader, RawCifar10Dataloader
from torch.utils.data import DataLoader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.plot.Plot import Plot, Figure
from Dataloader import Cifar10Dataloader, RawCifar10Dataloader
from torch.utils.data import DataLoader
import torch
import time
from optimization_playground_shared.models.BasicConvModel import BasicConvModel

device =  'cuda:0' if torch.cuda.is_available() else 'cpu' 

dataset = Cifar10Dataloader()

train_loader = DataLoader(dataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=2)
raw_train_loader = DataLoader(RawCifar10Dataloader(dataset.dataset),
                          batch_size=256,
                          shuffle=True,
                          num_workers=2)

reference_model = BasicConvModel(
    input_shape=((3, 32, 32))
).to(device)

fixmatch_model = FixMatchModel(BasicConvModel(
    input_shape=((3, 32, 32))
).to(device))
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

start = time.time()
for epoch in range(1_500):
    fixxmatch_accuracy = 0
    samples = 0
    loss_training = 0
    for (x, y, z) in train_loader:
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        fixmatch_model_optimizer.zero_grad()
        (loss, accuracy) = fixmatch_model.training_step((x, y,z))
        loss.backward()
        fixmatch_model_optimizer.step()
        loss_training += loss.item()
        # accuracy
        fixxmatch_accuracy += accuracy
        samples += 1
        assert fixxmatch_accuracy <= samples, f"{fixxmatch_accuracy} vs {samples}"

    epochs_second = (epoch + 1) / (time.time() - start)
    print(f"Epoch {epoch}, epochs / second {epochs_second}")

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
                    "Training loss with FixMatch": training_loss_mixmatch,
                    "Training loss without FixMatch": training_loss_reference,
                },
                title="Loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Training accuracy with FixMatch": training_accuracy_mixmatch,
                    "Training accuracy without FixMatch": training_accuracy_reference,
                },
                title="Training accuracy",
                x_axes_text="Epoch",
                y_axes_text="Accuracy",
            ),
            Figure(
                plots={
                    "Test accuracy with FixMatch": test_accuracy_mixmatch,
                    "Test accuracy without FixMatch": test_accuracy_reference,
                },
                title="Test accuracy",
                x_axes_text="Epoch",
                y_axes_text="Accuracy",
            )
        ],
        name='results.png'
    )

