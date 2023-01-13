from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.logging.EpochRuns import EpochRuns

gpt__like = EpochRuns(
    "gpt-like"
)
linear_model = EpochRuns(
    "non_attention_linear"
)
classical_attention = EpochRuns(
    "classical_attention"
)

plot = SimplePlot()

plot.plot(
    [
        LinePlot(y=gpt__like.fetch("train_loss"), legend="Gpt attention", title="Compared loss", x_text="Epochs", y_text="Loss"),
        LinePlot(y=linear_model.fetch("train_loss"), legend="Linear layer"),
        LinePlot(y=classical_attention.fetch("train_loss"), legend="classical attention layer")
    ]
)

plot.save("loss.png")
