from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
import os

def load_accuracy(file_name, **kwargs):
    train_accuracy = list(map(float, filter(lambda x: len(x) > 0, open(file_name, "r").read().split("\n"))))
    return LinePlot(y=train_accuracy, **kwargs)

def plot_it():
    plot = SimplePlot()
    path = lambda file: os.path.join(
        os.path.dirname(__file__),
        file
    )
    plot.plot(
        [
            load_accuracy(path("train_model_with_byol_features.txt"), title="Test accuracy", legend="Byol + predictor", x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100),
            load_accuracy(path("train_model_without_features.txt"), title="Test accuracy", legend="'Plain' model", x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100),
            load_accuracy(path("train_model_with_random_features.txt"), title="Test accuracy", legend="random features + predictor" , x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100)
        ]
    )
    plot.save(
        path("accuracy.png")
    )

    plot = SimplePlot()
    path = lambda file: os.path.join(
        os.path.dirname(__file__),
        file
    )
    plot.plot(
        [
            load_accuracy(path("byol_loss.txt"), title="Byol loss"),
        ]
    )
    plot.save(
        path("loss_byol.png")
    )
    print("created plot")

if __name__ == "__main__":
    plot_it()
