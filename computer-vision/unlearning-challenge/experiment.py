from optimization_playground_shared.dataloaders.Cifar100 import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt

baseline_train, baseline_test = get_dataloader()

forget_train, forget_test = get_dataloader(
    remove_classes=["motorcycle"]
)

forget_eval, _ = get_dataloader(
    get_classes=["motorcycle"]
)

model = BasicConvModel(
    input_shape=(3, 32, 32),
    num_classes=100
)
optimizer = optim.Adam(model.parameters())
iterator = TrainingLoop(model, optimizer)


print("Training baseline")
for i in range(10):
    (loss, acc) = iterator.train(baseline_train)
accuracy = iterator.eval(baseline_test)
print(f"Got {accuracy}% accuracy on baseline test")

accuracy_on_forgetting = iterator.eval(forget_eval)
print(f"Got {accuracy_on_forgetting}% accuracy on eval test (how good accuracy does it have on classes it should forget)")
print("")
print("")

print("Training to forget")
for i in range(10):
    (loss, acc) = iterator.train(forget_train)
    accuracy = iterator.eval(forget_test)
    print(f"Got {accuracy}% accuracy on forget test")

accuracy_on_forgetting = iterator.eval(forget_eval)
print(f"Got {accuracy_on_forgetting}% accuracy on eval test (how good accuracy does it have on classes it should forget)")
