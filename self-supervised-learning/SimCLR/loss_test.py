import unittest
import torch
from loss import Loss

class TestStringMethods(unittest.TestCase):

    def test_sim_should_be_equal(self):
        obj = Loss()
        N = 10
        vector = torch.rand(N * 2, 10)
        assert torch.allclose(
            obj.fast_sim(vector),
            obj.slow_sim(vector, N)
        )

    def test_fast_merge(self):
        obj = Loss()
        N = 10
        vector_1 = torch.rand(N, 10)
        vector_2 = torch.rand(N, 10)
        
        assert torch.allclose(
            obj.fast_combine(vector_1, vector_2),
            obj.slow_combine(vector_1, vector_2, N)
        )

    def test_fast_loss(self):
        obj = Loss()
        N = 10
        vector_1 = torch.rand(N, 10)
        vector_2 = torch.rand(N, 10)
        
        assert len(obj.loss(vector_1, vector_2).shape) == 0
        assert type(obj.loss(vector_1, vector_2).item()) == float

    def test_slow_loss(self):
        obj = Loss()
        N = 64
        vector_1 = torch.rand(N, 10)
        vector_2 = torch.rand(N, 10)
        
        assert len(obj.slow_loss(vector_1, vector_2, N).shape) == 0

    def test_slow_loss(self):
        obj = Loss()
        N = 64
        vector_1 = torch.rand(N, 10)
        vector_2 = torch.rand(N, 10)
        print(obj.loss(vector_1, vector_2))
        print(obj.slow_loss(vector_1, vector_2))

        assert torch.allclose(
            obj.loss(vector_1, vector_2),
            obj.slow_loss(vector_1, vector_2)
        )
if __name__ == '__main__':
    unittest.main()

