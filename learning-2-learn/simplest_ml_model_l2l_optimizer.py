from pickletools import optimize
from simplest_ml_model import train
from model_torch_optimizer import TorchLearning2LearnOptimizer

if __name__ == "__main__":
    optimizer = TorchLearning2LearnOptimizer
    train(
        optimizer,
        lr=1,
		max_epochs=100
    )
