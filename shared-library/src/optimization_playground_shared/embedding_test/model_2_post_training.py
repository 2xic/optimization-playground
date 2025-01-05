from optimization_playground_shared.distributed.MultipleGpuTrainWrapper import MultipleGpuTrainWrapper
from .model_config import ModelConfig
from .model_variants import GptEmbeddingsFineTuned
from .evals import EvaluationMetrics
import torch

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig("NextTokenPrediction_post_training")
        # TODO: this isn't a clean way at all to load in a moodel.
        old_name = self.model_config._model.name
        self.model_config._model.name = "NextTokenPrediction"
        self.model_config.load_trained_model()
        self.model_config._model.name = old_name
        assert self.model_config._model.document_encoder is not None

    def get_training_parameters(self):
        return self.model_config.get_model_parameters()

    def train(self, device):
        self.model_config.train(device)

class TrainerB(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig("NextTokenPrediction_post_training")
        # TODO: this isn't a clean way at all to load in a model.
        old_name = self.model_config._model.name
        self.model_config._model.name = "NextTokenPrediction"
        self.model_config.load_trained_model()
        self.model_config._model.name = old_name
        assert self.model_config._model.document_encoder is not None
        # Load in the new model.
        self.model_config._model.fine_tune_model(
            GptEmbeddingsFineTuned(self.model_config._model.config)
        )

    def fine_tune_model(self):
        evaluationMetrics = EvaluationMetrics()
        device = torch.device('cuda:0')
        self.model_config._model.model.to(device)
#        for (X, y) in zip(evaluationMetrics.X_train_original, evaluationMetrics.y_train_original):
        X = evaluationMetrics.X_train_original
        y = evaluationMetrics.y_train_original
        batch_size = 32
        self.model_config._model._device = device
        optimizer = torch.optim.Adam(self.model_config._model.model.parameters())
        for i in range(100):
            for _ in range(0, len(X), batch_size):
                embeddings_x = self.model_config._model.model.get_embedding(X[i:i+batch_size], self.model_config._model.document_encoder, device=device)
                error = self.model_config._model.model.linear_layer(
                    embeddings_x
                )
                loss = torch.nn.functional.binary_cross_entropy(
                    error,
                    torch.tensor(y[i:i+batch_size]).reshape(error.shape).float().to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
                # Error  
        
        print(evaluationMetrics.eval(self.model_config._model))


    def get_training_parameters(self):
        return self.model_config.get_model_parameters()

    def train(self, device):
        self.model_config.train(device)

if __name__ == "__main__":
#    trainer = Trainer()
#    trainer.start(is_debug_mode=True)
    trainer = TrainerB()
    trainer.fine_tune_model()

