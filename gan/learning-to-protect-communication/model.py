from torch import nn
import torch

class BaseCommunicatorModel(nn.Module):
    def __init__(self, plaintext, sharedkey, input_shape=None):
        super().__init__()
        self.plaintext = plaintext
        self.sharedkey = sharedkey
        self.input_shape = (plaintext + sharedkey) if input_shape is None else input_shape
        self.conv1_layout = [
            (4, 1, 2, 1, 0),
            (2, 2, 4, 2, 0),
            (1, 4, 4, 1, 0),
            (1, 4, 1, 1, 1),
        ]
        self.out = nn.Sequential(
            nn.Linear(self.input_shape, 2 * self.plaintext),
            # should this sigmoid be here ? 
            # The paper says "We use a sigmoid nonlinear unit after each layer except the final one"
          #  nn.Tanh(),
            *[
                self.create_conv_layer(*i)
                for i in self.conv1_layout
            ]
        )

    def create_conv_layer(self, kernel_size, input_depth,  output_depth, stride, padding=0):
        return nn.Sequential(
            nn.Sigmoid(),
            nn.Conv1d(
                input_depth,
                output_depth,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
          #  nn.Tanh()
        )

    def forward(self, x, shared_key):
        if shared_key is not None:
            x = torch.concat((
                x, 
                shared_key
            ), dim=1)
            assert x.shape[1] == self.sharedkey + self.plaintext
        #print(x.shape)
        x = x.reshape((x.shape[0], 1, x.shape[-1]))
        #print(x.shape)
        x = self.out(x)
        #print(x.shape)
        return nn.Tanh()(x.reshape((x.shape[0], -1)))

class Alice(BaseCommunicatorModel):
    def __init__(self, plaintext, sharedkey):
        super().__init__(
            plaintext=plaintext,
            sharedkey=sharedkey
        )

class Bob(BaseCommunicatorModel):
    def __init__(self, plaintext, sharedkey):
        super().__init__(
            plaintext=plaintext,
            sharedkey=sharedkey
        )

class Eve(BaseCommunicatorModel):
    def __init__(self, plaintext):
        super().__init__(
            plaintext=plaintext,
            sharedkey=0,
            input_shape=plaintext, #(plaintext * 2)
        )
       # print(self.input_shape)

    def forward(self, x):
   #     print(self)
        return super().forward(
            x,
            None
        )