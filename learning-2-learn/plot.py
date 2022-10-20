from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.logging.EpochRuns import EpochRuns

l2l_untrained = EpochRuns(
    "l2l"
)
l2l_trained = EpochRuns(
    "l2ltrained"
)
adam = EpochRuns(
    "adam"
)

plot = SimplePlot()

plot.plot(
    [
        LinePlot(y=adam.fetch("x"), legend="Adam", title="Value of x", x_text="Every 5 Epochs", y_text="X", y_min=-5, y_max=5),
        LinePlot(y=l2l_trained.fetch("x"), legend="L2l(trained"),
        LinePlot(y=l2l_untrained.fetch("x"), legend="L2l(untrained)")
    ]
)

plot.save("optimizer_test.png")
