import unittest
import numpy as np
import torch
from attention import AttentionLayer

class ModelForTest(torch.nn.Module):
    def __init__(self):
        super(ModelForTest, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.attention = AttentionLayer(
            1, 200
        )

    def create_block(self):
        return (
            AttentionLayer(300, 300),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.attention(x)
        return x


class TestLayerForward(unittest.TestCase):
    def test_forward(self):
        shape = (ModelForTest()(torch.zeros(1, 100)).shape)
        assert shape[0] == 1
        assert shape[1] == 200

if __name__ == '__main__':
    unittest.main()
