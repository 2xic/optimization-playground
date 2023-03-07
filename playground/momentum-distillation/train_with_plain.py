from optimization_playground_shared.models.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.optim as optim
import matplotlib.pyplot as plt

x = []
y_train_accuracy, y_test_accuracy = [], []

model = BasicConvModel(input_shape=(3, 32, 32))
train, test = get_dataloader()

optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)
epochs = 1_00

for epoch in range(epochs):
    (loss, acc) = iterator.train(train)
    print(({
        "epoch": epoch,
        "loss": loss,
        "acc": acc,
    }))

    if epoch % 10 == 0:
        y_train_accuracy.append(acc.item())
        accuracy = iterator.eval(test)
        y_test_accuracy.append(accuracy.item())
        x.append(epoch)

plt.title('Accuracy training dataset on CIFAR-10')
plt.plot(x, y_train_accuracy)
# plt.xscale('symlog')
plt.savefig('./plain/training.png')
plt.clf()

plt.title('Accuracy on test dataset on CIFAR-10')
plt.plot(x, y_test_accuracy)
# plt.xscale('symlog')
plt.savefig('./plain/testing.png')
