from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import get_dataloader

y_train_acc = []
y_test_acc = []

train, test = get_dataloader()
model = BasicConvModel(input_shape=(1, 28, 28 * 2))
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)

for epoch in range(10):
    print(f"{epoch}")
    (loss, acc) = iterator.train(train)
    accuracy = iterator.eval(test)
    y_train_acc.append(acc)
    y_test_acc.append(accuracy)

plt.clf()
plt.plot(y_train_acc, label="train")
plt.plot(y_test_acc, label="test")
plt.xlabel(f'Epochs')
plt.ylabel('Accuracy %')
plt.legend(loc="upper left")
plt.savefig(f'results.png')
plt.clf()
