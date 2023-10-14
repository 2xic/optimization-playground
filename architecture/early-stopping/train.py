from optimization_playground_shared.dataloaders.Mnist import get_dataloader_validation
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
from optimization_playground_shared.plot.Plot import Plot, Figure

class EpochTracking:
    def __init__(self) -> None:
        self.y = []
        self.x = []

    def add_item(self, value):
        self.x.append(len(self.x))
        self.y.append(value)

train_accuracy = EpochTracking()
test_accuracy = EpochTracking()
validation_accuracy = EpochTracking()
train_loss = EpochTracking()
test_loss = EpochTracking()
validation_loss = EpochTracking()


train, test, validation = get_dataloader_validation(subset=10_000)

model = BasicConvModel()
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)
best_validation_loss = float('inf')
# if the validation doesn't improve for these number of steps we can stop
n_step_validation = 5
n_best_loss_index = -1

for epoch in range(30):
    (loss, acc) = iterator.train(train)
    print(f"epoch: {epoch}, loss: {loss}, best_val loss {best_validation_loss} at epoch {n_best_loss_index}")
    train_accuracy.add_item(acc)
    train_loss.add_item(loss)
    # test
    (loss, accuracy) = iterator.eval_with_loss(test)
    test_accuracy.add_item(accuracy)
    test_loss.add_item(loss)
    # validation
    (loss, accuracy) = iterator.eval_with_loss(validation)
    validation_accuracy.add_item(accuracy)
    validation_loss.add_item(loss)
    if loss.item() < best_validation_loss:
        # we want to have a actual change in the loss
        if 1e-1 < (best_validation_loss - loss.item()):
            best_validation_loss = loss.item()
            n_best_loss_index = epoch
    elif n_step_validation <= (epoch - n_best_loss_index):
        print("We have moved past the optimal space ... we could stop here")
    
plot = Plot()
plot.plot_figures(
    figures=[
        Figure(
            plots={
                "Training": train_accuracy.y,
                "Testing": test_accuracy.y,
                "Validation": validation_accuracy.y,
            },
            title="Accuracy",
            x_axes_text="Epochs",
            y_axes_text="Accuracy",
        ),
        Figure(
            plots={
                "Training": train_loss.y,
                "Testing": test_loss.y,
                "Validation": validation_loss.y,
            },
            title="Loss",
            x_axes_text="Epochs",
            y_axes_text="Loss",
        ),
        Figure(
            plots={
                "Testing": test_loss.y,
                "Validation": validation_loss.y,
            },
            title="Loss for testing and validation",
            x_axes_text="Epochs",
            y_axes_text="Loss",
        )
    ],
    name='training.png'
)
