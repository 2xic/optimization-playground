"""
Training on one GPU takes a long time
"""
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import json

y_train_accuracy, y_test_accuracy = [], []

train, test = get_dataloader()

model = BasicConvModel(n_channels=3)
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)

for epoch in range(10_000):
    (loss, acc) = iterator.train(train)

    if epoch % 10 == 0:
        print(epoch)
        
        y_train_accuracy.append(acc.item())
        accuracy = iterator.eval(test)
        y_test_accuracy.append(accuracy.item())
        
        with open("results.json", "w") as file:
            file.write(json.dumps({
                "training": y_train_accuracy,
                "testing": y_test_accuracy
            }))
