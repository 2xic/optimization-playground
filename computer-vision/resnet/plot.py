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

loss_plot = SimplePlot()
min_y_loss = min(min(resnet.fetch("train_loss")), min(non_resnet.fetch("train_loss")))
max_y_loss = max(max(resnet.fetch("train_loss")), max(non_resnet.fetch("train_loss")))
loss_plot.plot(
    [
        LinePlot(y=resnet.fetch("train_loss"), legend="Resnet", title="Compared loss", x_text="Epochs", y_text="Loss",  ),
        LinePlot(y=non_resnet.fetch("train_loss"), legend="Non Resnet", y_min=min_y_loss, y_max=max_y_loss)
    ]
)
loss_plot.save("loss_resnet_vs_non_resnet.png")

training_accuracy = SimplePlot()
training_accuracy.plot(
    [
        LinePlot(y=resnet.fetch("training_accuracy"), legend="Resnet", title="Compared training accuracy", x_text="Epochs", y_text="Accuracy %", y_min=0, y_max=100),
        LinePlot(y=non_resnet.fetch("training_accuracy"), legend="Non Resnet")
    ]
)
training_accuracy.save("training_accuracy_resnet_vs_non_resnet.png")

testing_accuracy = SimplePlot()
testing_accuracy.plot(
    [
        LinePlot(y=resnet.fetch("testing_accuracy"), legend="Resnet", title="Compared testing accuracy", x_text="Epochs", y_text="Accuracy %", y_min=0, y_max=100),
        LinePlot(y=non_resnet.fetch("testing_accuracy"), legend="Non Resnet")
    ]
)
testing_accuracy.save("testing_accuracy_resnet_vs_non_resnet.png")
