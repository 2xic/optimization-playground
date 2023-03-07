from optimization_playground_shared.dataloaders.CorrupedMnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.models.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import test_dataloader, train_dataloader

train, test = train_dataloader, test_dataloader
model = BasicConvModel(input_shape=(4, 600, 800))
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)

epoch_accuracy = []
for _ in range(10):
    (loss, acc) = iterator.train(train)
    print(loss)
    
    accuracy = iterator.eval(test)
    epoch_accuracy.append(accuracy)

plt.clf()
plt.plot(epoch_accuracy)
plt.xlabel(f'Epcohs')
plt.ylabel('Accuracy on test dataset %')
plt.savefig(f'accuracy.png')
plt.clf()
