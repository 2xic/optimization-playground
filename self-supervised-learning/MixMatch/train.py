from model import Net
import torch
from sharpen import sharpen
from mixup import MixUp
from augmentations import augmentations
from hyperparameters import T, lambda_value
import torch
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.plot.Plot import Plot, Figure
from Dataloader import Cifar10Dataloader, RawCifar10Dataloader
from torch.utils.data import DataLoader
import time

"""
1. Run augmentation on a batch
    - Add a test util to the optimization library maybe
2. Feed the model the augmentations
    - Get the average predictions.
    - Sharpen it
3. Create a batch W of labeled and unlabeled batch
    - MixUp labeled batch
    - MixUp unlabeled batch
4. Loss is applied to the Created batches weighted on lambda.
    - Cross entropy on labeled batch
    - Squared l2 loss on unlabeled batch
"""

model = Net()
device =  'cuda:0' if torch.cuda.is_available() else 'cpu' 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.to(device)
step_size = 4
batch_size = 64

# Reference model with no mixup
reference_model = Net().to(device)
reference_optimizer = torch.optim.Adam(reference_model.parameters())
reference_loop = TrainingLoop(reference_model, reference_optimizer, loss=torch.nn.CrossEntropyLoss())

# Only 1000 labels
_, test_loader = get_dataloader(
    batch_size=batch_size
)
custom_cifar = Cifar10Dataloader()
train_dataloader = DataLoader(
    custom_cifar,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
train_raw = DataLoader(
    RawCifar10Dataloader(custom_cifar.dataset),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

training_loss_mixmatch = []
training_loss_reference = []

training_accuracy_mixmatch = []
training_accuracy_reference = []

test_accuracy_mixmatch = []
test_accuracy_reference = []
start = time.time()

for epoch in range(1_500):
    print(f"Epoch {epoch}")
    loss_training = 0
    mixmatch_accuracy = 0
    samples = 0
    for index, (x, y, unlabeled) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        unlabeled = unlabeled.to(device)

        x_augmentation = augmentations(x).float()
        """
        TODO: This could be more tensor effective
        """
        aug = augmentations(unlabeled)
        with torch.no_grad():
            aug_label = sharpen(model(aug), T=T)
        
        split = batch_size // 2

        (X_mixed, y_mixed) = MixUp()(x_augmentation[:split], y[:split], aug[:split], aug_label[:split], device)
        (u_mixed, uy_mixed) = MixUp()(x_augmentation[split:], y[split:], aug[split:], aug_label[split:], device)

        loss_labeled = torch.tensor(0.0, dtype=torch.float, device=device)
        x_predicted = model(X_mixed)
        y_max = torch.argmax(y_mixed, dim=1).reshape((y_mixed.shape[0]))
        loss_labeled += torch.nn.CrossEntropyLoss()(x_predicted, y_max)

        x_predicted = model(u_mixed)
        loss_unlabeled = torch.nn.MSELoss()(x_predicted, uy_mixed)
        if torch.isnan(loss_unlabeled):
            #print("loss_unlabeled is nan")
            continue

        optimizer.zero_grad(
            set_to_none=True
        )
        loss = loss_labeled + loss_unlabeled * lambda_value
        assert not torch.isnan(loss_labeled), "loss_labeled is nan"
        assert not torch.isnan(loss_unlabeled), "loss_unlabeled is nan"
        assert not torch.isnan(lambda_value), "lambda_value is nan"
        assert not torch.isnan(loss)
        loss.backward()

        loss_training += loss.item()
        assert x.shape[0] == y.shape[0]
        mixmatch_accuracy += (torch.argmax(model(x), dim=1) == torch.argmax(y, dim=1)).sum().item()
        samples += y.shape[0]
        assert mixmatch_accuracy <= samples, f"{mixmatch_accuracy} vs {samples}"
            
        optimizer.step()
    
    # one epoch
    epochs_second = (epoch + 1) / (time.time() - start)
    print(f"Epoch {epoch}, epochs / second {epochs_second}")

    #accuracy = reference_loop.train(test_loader)
    training_loss, training_accuracy = reference_loop.train(train_raw)
    test_accuracy = reference_loop.eval(test_loader)
    training_accuracy_mixmatch.append(
        (mixmatch_accuracy / samples) * 100
    )
    assert training_accuracy_mixmatch[0] <= 100, f"{samples} vs {mixmatch_accuracy}"
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
        TrainingLoop(model, None, None).eval(test_loader)
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

