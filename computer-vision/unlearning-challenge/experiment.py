from optimization_playground_shared.dataloaders.Cifar100 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
from optimization_playground_shared.plot.Plot import Plot, Figure

classes_to_forget = [
    "motorcycle", 
    "beaver",
    "clock",
    "telephone",
    "crocodile", 
    "dinosaur", 
    "lizard", 
    "snake", 
    "turtle"
]

baseline_train, baseline_test = get_dataloader()

forget_train, forget_test = get_dataloader(
    remove_classes=classes_to_forget
)

forget_eval, _ = get_dataloader(
    get_classes=classes_to_forget
)

model = BasicConvModel(
    input_shape=(3, 32, 32),
    num_classes=100
)
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)

EPOCHS_TRAIN = 1_00
EPOCHS_FORGET = 5

def plot_training(acc, classes_to_forget_later_acc, training_loss, training_classes_loss):
    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Training accuracy": acc,
                    "Classes to forget": classes_to_forget_later_acc,
                },
                title="Training accuracy",
                x_axes_text="Epochs",
                y_axes_text="Accuracy",
                y_min=0,
                y_max=100
            ),
            Figure(
                plots={
                    "Loss training": training_loss,
                    "Loss on class to forget": training_classes_loss,
                },
                title="Loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            )
        ],
        name='training.png'
    )

def plot_forgetting(acc, forgetting_classes_acc, loss, forgetting_classes_loss):
    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Training accuracy": acc,
                    "Classes to forget": forgetting_classes_acc,
                },
                title="Training accuracy",
                x_axes_text="Epochs",
                y_axes_text="Accuracy",
                y_min=0,
                y_max=100
            ),
            Figure(
                plots={
                    "Loss forgetting": loss,
                    "Loss to forget": forgetting_classes_loss,
                },
                title="Loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            )
        ],
        name='forgetting.png'
    )

def train_baseline():
    training_acc = []
    classes_to_forget_later_acc = []
    training_loss = []
    training_classes_loss = []
    
    for i in range(EPOCHS_TRAIN):
        (loss, acc) = iterator.train(baseline_train)
        (loss_forgetting, accuracy_on_forgetting) = iterator.eval_with_loss(forget_eval)

        training_acc.append(acc)
        classes_to_forget_later_acc.append(accuracy_on_forgetting)
        training_loss.append(loss)
        training_classes_loss.append(loss_forgetting)

    plot_training(
        training_acc,
        classes_to_forget_later_acc,
        training_loss,
        training_classes_loss
    )

def train_forgetting():
    print("Training to forget")
    forgetting_acc = []
    forgetting_classes_acc = []
    forgetting_loss = []
    forgetting_classes_loss = []
    
    for i in range(EPOCHS_FORGET):
        (loss, acc) = iterator.train(forget_train)
        (loss_forgetting, accuracy_on_forgetting) = iterator.eval_with_loss(forget_eval)
        forgetting_acc.append(acc)
        forgetting_classes_acc.append(accuracy_on_forgetting)
        forgetting_loss.append(loss)
        forgetting_classes_loss.append(loss_forgetting)

    plot_forgetting(
        forgetting_acc,
        forgetting_classes_acc,
        forgetting_loss,
        forgetting_classes_loss
    )

if __name__ == "__main__":
    train_baseline()
    train_forgetting()
