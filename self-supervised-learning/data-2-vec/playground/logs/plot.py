from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
import os

def load_accuracy(file_name, text):
    train_accuracy = list(map(float, filter(lambda x: len(x) > 0, open(file_name, "r").read().split("\n"))))
    return LinePlot(y=train_accuracy, title="Test accuracy", legend=text, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100)

def plot_it():
    plot = SimplePlot()
    path = lambda file: os.path.join(
        os.path.dirname(__file__),
        file
    )
    plot.plot(
        [
            load_accuracy(path("train_model_with_byol_features.txt"), text="Byol + predictor"),
            load_accuracy(path("train_model_without_features.txt"), text="'Plain' model"),
            load_accuracy(path("train_model_with_random_features.txt"), text="random features + predictor")
        ]
    )

    plot.save(
        path("accuracy.png")
    )
    print("created plot")

if __name__ == "__main__":
    plot_it()
