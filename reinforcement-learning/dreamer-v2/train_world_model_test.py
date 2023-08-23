import unittest
import torch
from train_world_model import RecurrentModel, WorldModel

class TestTrainModel(unittest.TestCase):
    def test_recurrent_model_shapes(self):
        latent = torch.zeros((4, 32))
        action = torch.zeros((4, 4))
        hidden = None

        recurrent_model = RecurrentModel()
        output, hidden = recurrent_model.forward(latent, action, hidden)

        assert len(output.shape) == 2
        assert output.shape[0] == 4
        
        # redoing it with updated hidden
        output, hidden = recurrent_model.forward(latent, action, hidden)
        assert len(output.shape) == 2
        assert output.shape[0] == 4
        
    def test_representation_model(self):
        image = torch.zeros((4, 1, 40, 40))
        
        latent = torch.zeros((4, 32))
        action = torch.zeros((4, 4))

        recurrent_model = RecurrentModel()
        _, hidden = recurrent_model.forward(latent, action, None)

        representation_model = WorldModel()
        new_latent = representation_model.representation(image, hidden)
        representation = representation_model.image_predictor(new_latent)
        
        assert representation.shape == image.shape

    def test_transition(self):
        latent = torch.zeros((4, 128))
        representation_model = WorldModel()
        new_latent = representation_model.transition(latent)
        
        assert new_latent.shape == latent.shape

if __name__ == '__main__':
    unittest.main()
