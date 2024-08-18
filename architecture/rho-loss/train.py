from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.nn as nn
import torch
from optimization_playground_shared.dataloaders.Mnist import get_dataloader_validation
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.training_loops.TrainingLoopPlot import TrainingLoopPlot
import torch.optim as optim
import random
import tqdm

# validation = holdout set ? 
train, test, holdout_set = get_dataloader_validation()

big_model = BasicConvModel()
big_model_optimizer = optim.Adam(big_model.parameters())

big_model_no_rho = BasicConvModel()
big_model_optimizer_no_rho = optim.Adam(big_model_no_rho.parameters())


small_model = BasicConvModel()
small_model.out = nn.Sequential(
    small_model.fc1,
    nn.ReLU(),
    nn.Linear(256, big_model.num_classes),
    nn.LogSoftmax(dim=1),
)

# train model on holdout set
optimizer = optim.Adam(small_model.parameters())
training_loop = TrainingLoop(
    model=small_model,
    optimizer=optimizer,
)
#print(holdout_set)
training_loop.train(holdout_set)

# iterate over the dataset
x_shape = len(train.dataset)
irreducibleLoss = torch.zeros((x_shape))
for index in range(irreducibleLoss.shape[0]):
    with torch.no_grad():
        x, y = train.dataset[index]
        irreducibleLoss[index] = torch.nn.functional.nll_loss(
            small_model(x.reshape((1, ) + x.shape)),
            torch.tensor([y]),
        )

# ^ generates the first parts
for _  in tqdm.tqdm(range(100)):
    # 1. select a large batch size
    # 2. compute the loss
    # 3. Diff the loss against irreducibleLoss to get the rho loss
    # 4. Use the minibatch of n_samples from rho loss to train the model
    # 5. step the optim model and win.
    
    batch_size_large = 1024
    random_offset = random.randint(0, x_shape - batch_size_large)
    
    x = torch.zeros((batch_size_large, 1, 28, 28))
    y = torch.zeros((batch_size_large))
    for i in range(random_offset, random_offset + batch_size_large):    
        _x, _y = train.dataset[i]
        x[i - random_offset] = _x
        y[i - random_offset] = _y

    loss = torch.nn.functional.nll_loss(
        big_model(x),
        y.long(),
        reduction='none'
    )
    rho_loss = loss - irreducibleLoss[random_offset:random_offset + batch_size_large]
    sorted_tensor, sorted_indices = torch.sort(rho_loss, dim=0)
    
    small_batch_size = 32
    new_batch_indexes = sorted_indices[-small_batch_size:]
    x = torch.zeros((small_batch_size, 1, 28, 28))
    y = torch.zeros((small_batch_size))
    for i in range(small_batch_size):
        _x, _y = train.dataset[new_batch_indexes[i]]
        x[i] = _x
        y[i] = _y

    loss = torch.nn.functional.nll_loss(
        big_model(x),
        y.long()
    )
    big_model_optimizer.zero_grad()
    loss.backward()
    big_model_optimizer.step()
    # train the other model on same size of something random 
    x = torch.zeros((small_batch_size, 1, 28, 28))
    y = torch.zeros((small_batch_size))
    for i in range(0, small_batch_size):
        _x, _y = train.dataset[random_offset + i]
        x[i] = _x
        y[i] = _y
    loss = torch.nn.functional.nll_loss(
        big_model_no_rho(x),
        y.long()
    )
    big_model_optimizer_no_rho.zero_grad()
    loss.backward()
    big_model_optimizer_no_rho.step()


print("RHO")
accuracy_eval = TrainingLoop(big_model, big_model_optimizer).eval(test).item()
print(f"\t{accuracy_eval}")
print("NO RHO")
accuracy_eval = TrainingLoop(big_model_no_rho, big_model_optimizer_no_rho).eval(test).item()
print(f"\t{accuracy_eval}")

