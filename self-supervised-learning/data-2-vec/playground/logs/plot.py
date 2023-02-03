from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
import matplotlib.pyplot as plt

def load_accuracy(file_name, text):
    train_accuracy = list(map(float, filter(lambda x: len(x) > 0, open(file_name, "r").read().split("\n"))))
    return LinePlot(y=train_accuracy, title="Accuracy", legend=text, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100)

plot = SimplePlot()
plot.plot(
    [
        load_accuracy("train_model_with_byol_features.txt", text="Byol + predictor"),
        load_accuracy("train_model_without_features.txt", text="'Plain' model"),
        load_accuracy("train_model_with_random_features.txt", text="random features + predictor")
    ]
)

plot.save("accuracy.png")
