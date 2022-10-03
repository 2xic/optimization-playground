import torch.nn as nn

loss_reduction = 'mean' # "sum" # "mean"

# reduction on the output, but outside the model
output_reduction = lambda x: nn.Softmax(dim=1)(x)

warm_epoch = 100

supervised_size_ratio = 1 #0.25

