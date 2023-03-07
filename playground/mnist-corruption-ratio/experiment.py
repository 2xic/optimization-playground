from optimization_playground_shared.dataloaders.CorrupedMnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.models.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt


for max_train_size in [1_000, 10_000]:
    x_corruption = []
    y_accuracy = []
    for corruption_rate in range(0, 101, 1):
        print(f"Training with {corruption_rate}% corruption")
        train, test = get_dataloader(corruption_rate / 100, max_train_size=max_train_size)

        model = BasicConvModel()
        optimizer = optim.Adam(model.parameters())
        iterator = TrainingLoop(model, optimizer)

        for _ in range(10):
            (loss, acc) = iterator.train(train)

        accuracy = iterator.eval(test)
        print(f"Got {accuracy}% accuracy on test")

        x_corruption.append(corruption_rate)
        y_accuracy.append(accuracy.item())
     #   break

    plt.clf()
    plt.plot(x_corruption, y_accuracy)
    plt.xlabel(f'Corruption % with {max_train_size} MNIST dataset while traning')
    plt.ylabel('Accuracy on test dataset %')
    plt.savefig(f'corruption_{max_train_size}.png')
    plt.clf()
