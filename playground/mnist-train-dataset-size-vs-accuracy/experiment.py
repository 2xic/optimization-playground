from optimization_playground_shared.dataloaders.RestrictedMnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.models.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt

x_entries = []
y_accuracy = []
for entries_per_class in [1, 10, 50, 100, 250, 500]:
    print(f"Training with {entries_per_class} entries per class")
    train, test = get_dataloader(entries_per_class)

    model = BasicConvModel()
    optimizer = optim.Adam(model.parameters())
    iterator = TrainingLoop(model, optimizer)

    for _ in range(10):
        (loss, acc) = iterator.train(train)

    accuracy = iterator.eval(test)
    print(f"Got {accuracy}% accuracy on test")

    x_entries.append(entries_per_class)
    y_accuracy.append(accuracy.item())

plt.clf()
plt.plot(x_entries, y_accuracy)
plt.xlabel(f'Number of entries per class')
plt.ylabel('(Test)Accuracy %')
plt.savefig(f'dataset_size_vs_test_accuracy.png')
plt.clf()
