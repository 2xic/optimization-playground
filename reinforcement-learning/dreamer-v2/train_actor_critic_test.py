import unittest
import torch
from train_actor_critic import HiddenLatentRepresentation

class TestActorCriticModel(unittest.TestCase):
    def test_recurrent_model_shapes(self):
        latent = torch.zeros((4, 32))
        hidden = torch.zeros((4, 4))

        recurrent_model = HiddenLatentRepresentation(
            32,
            4
        )
        hidden = recurrent_model.forward(latent, hidden)
        assert hidden is not None

if __name__ == '__main__':
    unittest.main()
