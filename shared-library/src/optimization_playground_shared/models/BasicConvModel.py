from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def max_pool_output(shape_in, pool: nn.MaxPool2d):
    resolve_arr_or_int = lambda x, i: x if type(x) == int  else x[i]

    h_out = math.floor((shape_in[-2] +  2 * resolve_arr_or_int(pool.padding, 0) - resolve_arr_or_int(pool.dilation, 0) * (resolve_arr_or_int(pool.kernel_size, 0) - 1) -1)/(
        resolve_arr_or_int(pool.stride, 0)
    ) + 1)
    w_out = math.floor((shape_in[-1] + 2 * resolve_arr_or_int(pool.padding, 0) - resolve_arr_or_int(pool.dilation, 0) * (resolve_arr_or_int(pool.kernel_size, 1) - 1) -1)/(
        resolve_arr_or_int(pool.stride, 1)
    ) + 1)

    return (
        h_out,
        w_out
    )

def conv_output_shape(shape_in, conv: nn.Conv2d):
    resolve_arr_or_int = lambda x, i: x if type(x) == int  else x[i]
    h_out = math.floor(((shape_in[-2] + 2 * resolve_arr_or_int(conv.padding, 0) - resolve_arr_or_int(conv.dilation, 0) * (resolve_arr_or_int(conv.kernel_size, 0) - 1) -1)/(
        resolve_arr_or_int(conv.stride, 0)
    )) + 1)
    w_out = math.floor(((shape_in[-1] + 2 * resolve_arr_or_int(conv.padding, 0) - resolve_arr_or_int(conv.dilation, 0) * (resolve_arr_or_int(conv.kernel_size, 1) - 1) -1)/(
        resolve_arr_or_int(conv.stride, 1)
    )) + 1)
    return (
        h_out,
        w_out
    )

def get_output_shape(shape_in, layers):
    shape = shape_in
    for i in layers:
      #  print(shape)
        if isinstance(i, nn.Conv2d):
            shape = conv_output_shape(shape, i)
        elif isinstance(i, nn.MaxPool2d):
            shape = conv_output_shape(shape, i)
        elif hasattr(i, 'conv_shape'):
            shape = conv_output_shape(shape, i)
        elif isinstance(i, nn.Tanh):
            continue
        elif isinstance(i, nn.ReLU):
            continue
        elif isinstance(i, nn.ELU):
            continue
        elif isinstance(i, nn.LeakyReLU):
            continue
        elif isinstance(i, nn.BatchNorm2d):
            continue
        else:
            raise Exception(f"Hm, unknown layer {i}")
    return shape

class BasicConvModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.m = nn.BatchNorm2d(input_shape[0])
        self.conv1 = nn.Conv2d(input_shape[0], 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.output_shape = ((
            get_output_shape(input_shape,
            [self.conv1, self.pool, self.conv2, self.pool])
        ))
        self.fc1 = nn.Linear(16 * self.output_shape[0] * self.output_shape[1], 256)
        self.out = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 63),
            nn.ReLU(),
            nn.Dropout(p=0.01),
            nn.Linear(63, num_classes),
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
