"""
Training on one GPU takes a long time
"""
#from optimization_playground_shared.dataloaders.MnistAugmentation import get_dataloader
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.models.ConvModel import ConvModel
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from training_loop import StudentTeacherLoop

STEP_SIZE = 10
EPOCHS = 101

teacher_y_train_accuracy, teacher_y_test_accuracy = [], []
train, test = get_dataloader(
    subset=5_000,
)

#teacher = BasicConvModel()
teacher = ConvModel(input_shape=(1, 28, 28), layers=[
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.01),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
])
teacher_optimizer = optim.Adam(teacher.parameters())
teacher_iterator = TrainingLoop(teacher, teacher_optimizer)
teacher_iterator.loss = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    (loss, acc) = teacher_iterator.train(train)

    if epoch % STEP_SIZE == 0:        
        teacher_y_train_accuracy.append(acc.item())
        accuracy = teacher_iterator.eval(test)
        teacher_y_test_accuracy.append(accuracy.item())
        print(f"Teacher, epoch: {epoch} acc: {acc}")
print("")

train, test = get_dataloader(
    subset=1_00,
)
def train_student(with_teacher=True):
    student = ConvModel(input_shape=(1, 28, 28), layers=[
        nn.Linear(256, 32),
        nn.Dropout(p=0.01),
        nn.Linear(32, 10),
        nn.Softmax(dim=1),
    ])
    student_optimizer = optim.Adam(student.parameters())
    student_iterator = None
    if with_teacher:
        print("Using teacher model")
        student_iterator = StudentTeacherLoop(student, teacher, student_optimizer)
    else:
        print("Using cross entropy single loss")
        student_iterator = TrainingLoop(student, student_optimizer)
        student_iterator.loss = nn.CrossEntropyLoss()

    student_y_train_accuracy, student_y_test_accuracy = [], []
    for epoch in range(EPOCHS):
        (_, acc) = student_iterator.train(train)

        if epoch % STEP_SIZE == 0:        
            student_y_train_accuracy.append(acc.item())
            accuracy = student_iterator.eval(test)
            student_y_test_accuracy.append(accuracy.item())
            print(f"Student (with_teacher={with_teacher}), epoch: {epoch} acc_train: {acc}, acc_test: {accuracy.item()}")
    print("")
    return student_y_train_accuracy, student_y_test_accuracy

student_teacher_y_train_accuracy, student_teacher_y_test_accuracy = train_student(with_teacher=True)
student_y_train_accuracy, student_y_test_accuracy = train_student(with_teacher=False)

X = list(range(0, EPOCHS, STEP_SIZE))

plt.title('Accuracy training dataset on MNIST')
plt.plot(X, teacher_y_train_accuracy, label="teacher")
plt.plot(X, student_y_train_accuracy, label="student (no teacher)")
plt.plot(X, student_teacher_y_train_accuracy, label="student (with teacher)")
plt.legend(loc="upper left")
plt.savefig('training.png')
plt.clf()

plt.title('Accuracy on test dataset on MNIST')
plt.plot(X, teacher_y_test_accuracy, label="teacher")
plt.plot(X, student_y_test_accuracy, label="student (no teacher)")
plt.plot(X, student_teacher_y_test_accuracy, label="student (with teacher)")
plt.legend(loc="upper left")
plt.savefig('testing.png')
