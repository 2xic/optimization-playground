from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
#from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.dataloaders.MnistAugmentation import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.optim as optim
from optim import ScheduleFreeOptimizer
import matplotlib.pyplot as plt
from schedulefree.adamw_schedulefree import AdamWScheduleFree

_, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,6))


loss_optimizer = [
    lambda x: ScheduleFreeOptimizer(x.parameters()),
    lambda x: optim.Adam(x.parameters()),
    # The official version
    lambda x: AdamWScheduleFree(x.parameters()),
]

for index, get_optimizer in enumerate(loss_optimizer):
    train, test = get_dataloader()
    model = BasicConvModel(input_shape=(1, 28, 28))
    optimizer = get_optimizer(model)
    iterator = TrainingLoop(model, optimizer)
    epochs = 1_00

    loss_epoch = []
    acc_epoch = []
    name = optimizer.__class__.__name__
    for i in range(5):
        (loss, acc) = iterator.train(train)
        loss_epoch.append(loss.item())
        acc_epoch.append(acc.item())
        print(name, loss_epoch[-1], acc_epoch[-1], i)
    print("")

    color = [
        "red",
        "blue",
        "orange"
    ][index]
    axes[0].plot(acc_epoch, label=name, color=color)
    axes[1].plot(loss_epoch, label=name, color=color)
    axes[0].legend(loc="upper left")


plt.savefig("results.png")
plt.clf()
