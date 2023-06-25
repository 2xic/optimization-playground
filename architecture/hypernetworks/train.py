from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from MainNet import MainNet
from HyperNetwork import HyperNetwork
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.optim as optim
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.logging.EpochRuns import EpochRuns

EPOCHS = 3


def train_hyper_network():
    hyper_net = HyperNetwork(
        z=64,
    )
    model = MainNet(
        (1, 28, 28),
        hyper_net=hyper_net
    )
    optimizer = optim.Adam(model.parameters())
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
    )

    (train, test) = get_dataloader()
    accuracy = []
    for epoch in range(EPOCHS):
        (loss, acc) = training_loop.train(
            train
        )
        test_eval = training_loop.eval(test).item()
        print(
            f"{epoch} HyperNetwork, loss: {loss} train_acc: {acc} test_acc: {test_eval}")
        accuracy.append(test_eval)
    return accuracy


def train_baseline():
    (train, test) = get_dataloader()
    model = BasicConvModel(
        (1, 28, 28)
    )
    optimizer = optim.Adam(model.parameters())
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
    )
    accuracy = []
    for epoch in range(EPOCHS):
        (loss, acc) = training_loop.train(
            train
        )
        test_eval = training_loop.eval(test).item()
        print(f"{epoch} baseline, loss: {loss} train_acc: {acc} test_acc: {test_eval}")
        accuracy.append(test_eval)
    return accuracy


if __name__ == "__main__":
    hyper_network = train_hyper_network()
    baseline = train_baseline()

    plot = SimplePlot()

    plot.plot(
        [
            LinePlot(y=hyper_network, legend="hyper network",
                     title="Test accuracy", y_text="Accuracy (%)", x_text="Epoch"),
            LinePlot(y=baseline, legend="non hyper network variant", y_max=100, y_min=0),
        ],
    )

    plot.save("results.png")
