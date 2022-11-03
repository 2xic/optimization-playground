from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.logging.EpochRuns import EpochRuns

train_accuracy = list(map(float, filter(lambda x: len(x) > 0, open("train_accuracy.txt", "r").read().split("\n"))))

plot = SimplePlot()
plot.plot(
    [
        LinePlot(y=train_accuracy, title="Train accuracy", x_text="Increment of 10x64 batches", y_text="Accuracy"),
    ]
)

plot.save("train_loss.png")
