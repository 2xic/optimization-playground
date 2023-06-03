from dataset import get_simple_dataset, get_mnist_dataloader
from model import *
import torch
from optimization_playground_shared.utils.SimpleAverage import SimpleAverage
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_simple():
    avg_accuracy = SimpleAverage()
    avg_loss = SimpleAverage()
    for _ in range(10):
        layers = []
        layers.append(FfLinear(3, 15))
        layers.append(FfLinear(15, 15))
        layers.append(FfLinear(15, 5))

        errors = []
        accuracy = []

        model = Model(layers)
        positive, negative = get_simple_dataset()
        for i in range(10_000):
            error = model.train(
                positive,
                negative
            )
            if i % 128 == 0:
                print(error)
            errors.append(error.item())
            with torch.no_grad():
                acc = (
                    (1 < model.forward(positive)).sum() +
                    (model.forward(negative) < 1).sum()
                ) / (positive.shape[0] + negative.shape[0]) * 100
                accuracy.append(acc.item())
        avg_accuracy.add(accuracy)
        avg_loss.add(errors)

    training_accuracy = SimplePlot()
    training_accuracy.plot(
        [
            LinePlot(y=avg_accuracy.res(), y_text="Accuracy"),
        ],
    )
    training_accuracy.plot([
        LinePlot(y=avg_loss.res(), x_text="Iteration", y_text="Loss"),
    ],
    )
    training_accuracy.save("toy_example.png")


def train_mnist():
    print(device)
    train_dataloader, test_loader = get_mnist_dataloader()
    layers = []
    layers.append(FfLinear(794, 512, lr=0.03))
    layers.append(FfLinear(512, 256, p=0.2, lr=0.03))
    layers.append(FfLinear(256, 512, lr=0.03))

    model = Model(layers).to(device)

   # avg_accuracy = SimpleAverage()
   # avg_loss = SimpleAverage()

    errors = []
    accuracy = []
    for epoch in range(1_000):
        test_acc = eval_model(model, test_loader).item()
        print(f"({epoch}) acc == {test_acc}%")

        sum_error = 0
        for i, (X, y) in enumerate(train_dataloader):
            X = X.reshape((X.shape[0], -1)).to(device)
            y = torch.nn.functional.one_hot(y, 10).to(device)

            z_positive = torch.concat((
                y,
                X,
            ), dim=1)

            delta = torch.nn.functional.one_hot(torch.randint(size=(
                y.shape[0], 1), low=0, high=10, device=device).reshape(-1, 1), 10).reshape(-1, 10)
            delta -= y.reshape((-1, 10))

            z_negative = torch.concat((
                delta,
                X,
            ), dim=1)

            error = model.train(
                z_positive,
                z_negative
            )
            if i % 128 == 0:
                print(f"\t{error}")
            sum_error += error.item()

        errors.append(sum_error)
        accuracy.append(test_acc)

        if 0 < epoch and epoch % 32 == 0:
            training_accuracy = SimplePlot()
            training_accuracy.plot(
                [
                    LinePlot(y=accuracy, y_text="Accuracy"),
                ],
            )
            training_accuracy.plot([
                LinePlot(y=errors, x_text="Iteration", y_text="Loss"),
            ])
            training_accuracy.save("mnist.png")


def eval_model(model, test_loader):
    acc = torch.tensor((0), device=device, dtype=torch.float)
    batch_test = 0
    for _, (X, y) in enumerate(test_loader):
        X = X.reshape((X.shape[0], -1)).to(device)
        y = y.to(device).float()
        predicted, predicted_raw = predict_mnist(model, X)
      #  print(predicted)
        acc += ((y == predicted).sum() / y.shape[0]) * 100
        batch_test += X.shape[0]
    print(predicted_raw[:2], predicted_raw.shape)
    print(predicted[:16], predicted.shape)
    print(y[:16], y.shape)
    return (acc / batch_test) * 100


def predict_mnist(model, X):
    output = []
    for class_id in range(10):
        y_vec = torch.nn.functional.one_hot(torch.tensor([
            class_id
            for _ in range(X.shape[0])
        ], device=device), 10).reshape((-1, 10))

        z = torch.concat((
            y_vec,
            X,
        ), dim=1)
        score_class = model.forward(z).reshape((-1, 1))
        output.append(score_class)

    output_tensor_x = torch.concat(output, dim=1)
    predicted_classes = torch.argmax(output_tensor_x, dim=1)
    return predicted_classes, output_tensor_x


if __name__ == "__main__":
    #    train_simple()
    train_mnist()
