"""
Training on one GPU takes a long time
"""
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.models.ConvModel import ConvModel
from optimization_playground_shared.models.TrainingLoop import TrainingLoop
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

STEP_SIZE = 10
EPOCHS = 100

teacher_y_train_accuracy, teacher_y_test_accuracy = [], []
train, test = get_dataloader()

teacher = BasicConvModel()
teacher_optimizer = optim.Adam(teacher.parameters())
teacher_iterator = TrainingLoop(teacher, teacher_optimizer)

for epoch in range(EPOCHS):
    (loss, acc) = teacher_iterator.train(train)

    if epoch % STEP_SIZE == 0:        
        teacher_y_train_accuracy.append(acc.item())
        accuracy = teacher_iterator.eval(test)
        teacher_y_test_accuracy.append(accuracy.item())
        print(f"Teacher: {epoch}")

student = ConvModel(input_shape=(1, 28, 28), layers=[
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 63),
    nn.ReLU(),
    nn.Dropout(p=0.01),
    nn.Linear(63, 10),
    nn.LogSoftmax(dim=1),
])
student_optimizer = optim.Adam(student.parameters())
student_iterator = TrainingLoop(student, teacher_optimizer)
student_y_train_accuracy, student_y_test_accuracy = [], []


for epoch in range(EPOCHS):
    """
    TODO: Student iterator has to support soft labels from teacher
        -> See equation 1 in the paper
    """
    (loss, acc) = student_iterator.train(train)

    if epoch % STEP_SIZE == 0:        
        student_y_train_accuracy.append(acc.item())
        accuracy = teacher_iterator.eval(test)
        student_y_test_accuracy.append(accuracy.item())
        print(f"Student: {epoch}")


X = list(range(0, EPOCHS, STEP_SIZE))

plt.title('Accuracy training dataset on MNIST')
plt.plot(X, teacher_y_train_accuracy, label="teacher")
plt.plot(X, student_y_train_accuracy, label="student")
plt.legend(loc="upper left")
plt.savefig('training.png')
plt.clf()

plt.title('Accuracy on test dataset on MNIST')
plt.plot(X, teacher_y_test_accuracy, label="teacher")
plt.plot(X, student_y_test_accuracy, label="student")
plt.legend(loc="upper left")
plt.xscale('symlog')
plt.savefig('testing.png')
