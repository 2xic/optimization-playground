"""
https://arxiv.org/abs/2301.13142
https://news.ycombinator.com/item?id=41153039
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Conference/2019.md

The trick seems to just make the sparsity part of the optimization loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.plot.Plot import Plot, Figure
from tqdm import tqdm


"""
======
"""

import torch.autograd as autograd
class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x
"""
======
"""

# equation 1
def q(x, b, e):
    return (2 ** e) * torch.min(
        torch.max(
            2**(-e) * x,
            -2 ** (b-1)
        ),
        2 ** (b - 1) - 1
    )

class CompressionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0) -> None:
        super(CompressionConv, self).__init__()
        self.padding = padding
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.normal_(self.weight)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels)) # in_channels, kernel_size, kernel_size)
        nn.init.normal_(self.bias)
        # parameters used to compress the network
        eps = -8.
        bits = 2.
        self.bit_depth = torch.nn.Parameter(torch.zeros((out_channels, 1, 1, 1)).fill_(bits))
        self.floating_point_exp = torch.zeros((out_channels, 1, 1, 1)).fill_(eps)
        self.ste = StraightThroughEstimator()

    def bits(self):
        return F.relu(self.bit_depth).sum() + torch.prod(torch.tensor(self.weight.shape[1:]))

    def forward(self, input):
        weights = q(
            self.weight,
            F.relu(self.bit_depth),
            self.floating_point_exp
        )
        # Make the gradients flow, STE
        compressed_weight = self.ste(weights)
        compressed_weight = 2 ** self.floating_point_exp * compressed_weight
        return F.conv2d(input, compressed_weight, self.bias, self.stride,
                        self.padding, dilation=1, groups=1)


# Copy of BasicConvModel
class BasicConvModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, use_compression=False):
        super().__init__()
        conv_layer = nn.Conv2d
        if use_compression:
            conv_layer = CompressionConv

        self.m = nn.BatchNorm2d(input_shape[0])
        self.conv1 = conv_layer(input_shape[0], 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer(12, 16, 5)
        # self.output_shape = ((
        #     get_output_shape(input_shape,
        #     [self.conv1, self.pool, self.conv2, self.pool])
        # ))
        self.num_classes = num_classes
        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.m(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = self.out(x)
        return x

train, test = get_dataloader()
model = BasicConvModel(
    use_compression=True
)
optimizer = optim.Adam(model.parameters())
main_loss = torch.nn.NLLLoss()

def get_qbits(model):
    bits = 0
    if isinstance(model.conv1, CompressionConv):
        bits = model.conv1.bits() + model.conv2.bits()
    return bits

param_count = sum(p.numel() for p in model.parameters())

def custom_loss(y_pred, y):
    # NOTE: Without the average the model will progress badly
    gamma = 0.05 # compression factor
    bits = get_qbits(model) / param_count
    #print(bits)
    return main_loss(y_pred, y) + gamma * bits

iterator = TrainingLoop(model, optimizer, loss=custom_loss).use_tqdm()
training_accuracy = []
training_loss = []
training_qbits = []
epochs = 1_000

for _ in tqdm(range(epochs)):
    (loss, accuracy) = iterator.train(train)
    training_accuracy.append(accuracy)
    training_loss.append(loss)
    training_qbits.append(get_qbits(model))
    print(get_qbits(model))

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Loss": training_loss,
                },
                title="Training loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Training accuracy": training_accuracy,
                },
                title="Accuracy",
                x_axes_text="Epochs",
                y_axes_text="accuracy",
            ),
            Figure(
                plots={
                    "qbits": training_qbits,
                },
                title="Qbits",
                x_axes_text="Epochs",
                y_axes_text="Qbits",
            ),
        ],
        name=f'training.png'
    )
