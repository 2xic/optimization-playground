import unittest
import torch
from model import DecoderModel, EncoderModel
from iterate import Iterator

class TestLayerForward(unittest.TestCase):
    def test_iterate(self):
        rows = 1
        input_size = 10
        encoder = EncoderModel(input_size)
        decoder = DecoderModel(input_size)
        iterate = Iterator(
            encoder,
            decoder,
            1,
            2
        )
        input_tensor = torch.zeros(rows, input_size, dtype=torch.long)
        output_tensor = torch.zeros(rows, input_size, dtype=torch.long)

        iterate.iterate(
            input_tensor,
            output_tensor,
        )

if __name__ == '__main__':
    unittest.main()
