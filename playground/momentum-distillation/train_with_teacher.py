from optimization_playground_shared.models.TrainingLoopGenerated import TrainingLoopGenerated
from optimization_playground_shared.utils.UpdateWeightsSchedule import UpdateWeightsSchedule
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

x = []
y_train_accuracy, y_test_accuracy = [], []


student_model = BasicConvModel(input_shape=(3, 32, 32))
teacher_model = BasicConvModel(input_shape=(3, 32, 32))


def generated_labels(X):
    with torch.no_grad():
        return teacher_model(X.reshape((1, ) + X.shape))

train, test = get_dataloader(
    generated_labels=generated_labels
)

optimizer = optim.Adam(student_model.parameters())
iterator = TrainingLoopGenerated(student_model, optimizer)
epochs = 1_00
weight_updater = UpdateWeightsSchedule(
    student_model,
    teacher_model,
    epochs=epochs
)

for epoch in range(epochs):
    (loss, acc) = iterator.train(train)
    print(({
        "epoch": epoch,
        "loss": loss,
        "acc": acc,
        "t": weight_updater.t
    }))

    if epoch % 10 == 0:
        y_train_accuracy.append(acc.item())
        accuracy = iterator.eval(test)
        y_test_accuracy.append(accuracy.item())
        x.append(epoch)
    weight_updater.update(epoch)

plt.title('Accuracy training dataset on CIFAR-10')
plt.plot(x, y_train_accuracy)
# plt.xscale('symlog')
plt.savefig('with_teacher/training.png')
plt.clf()

plt.title('Accuracy on test dataset on CIFAR-10')
plt.plot(x, y_test_accuracy)
# plt.xscale('symlog')
plt.savefig('with_teacher/testing.png')
