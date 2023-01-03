from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.logging.EpochRuns import EpochRuns

resnet = EpochRuns(
    "resnet"
)
non_resnet = EpochRuns(
    "non_resnet"
)

# print(non_resnet.information)
# print(resnet.information)

plot = SimplePlot()

plot.plot(
    [
        LinePlot(y=resnet.fetch("train_loss"), legend="Resnet", title="Compared loss", x_text="Epochs", y_text="Loss"),
        LinePlot(y=non_resnet.fetch("train_loss"), legend="Non Resnet")
    ]
)

plot.save("resnet_vs_non_resnet.png")
