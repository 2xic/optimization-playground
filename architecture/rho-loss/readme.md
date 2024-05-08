### [Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://arxiv.org/pdf/2206.07137)
[Source code](https://github.com/OATML/RHO-Loss)

The idea is to compute how much information there is in a given batch of the dataset. How is this done ? You calculate the loss between the batch against some model that is also trained on some holdout set. This way to get some information about how useful that batch of data is.

````
Phase 1
# 0. Train a small model on a holdout dataset 
# 1. Run this model on the training set and compute the loss (irreducibleLoss)

Phase 2
# 1. select a large batch size and compute the loss with our actual model we want to train
# 2. Diff the loss against irreducibleLoss to get the rho loss
# 3. Sort this loss and train on the samples with larges loss (on a small batch size)
# 4. step the optim model and win.
```