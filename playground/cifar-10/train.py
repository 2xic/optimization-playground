from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
from models.PlainModel import PlainModel
from models.PlainModelDropout import PlainModelDropout
from models.PlainModelBatchNormAugmentation import PlainModelBatchNormAugmentation, PlainModelBatchNormAugmentationDropout
from models.PlainModelBatchNorm import PlainModelBatchNorm
from models.PlainModelResidual import PlainModelResidual
from models.PlainModelBatchNormDynamicLr import PlainModelBatchNormDynamicLr
from eval import eval_model
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_playground_shared.process_pools.ProcessPool import ProcessPool

EPOCHS = 100
BATCH_SIZE = 256

"""
Training logic
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
training_plots = []
test_plots = []

def evaluate(lock, model):
    dataloader = DataLoader(
        Cifar10Dataloader(),
        batch_size=256,
        shuffle=True,
    )
    X = []
    y = []
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )
    test_accuracy = []
    training_accuracy = []
    if hasattr(model, 'set_optimizer'):
        model.set_optimizer(optimizer)
    for epoch in range(EPOCHS):
        total_loss = torch.tensor(0.0, device=device)
        acc = torch.tensor(0.0, device=device)
        rows = 0
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)

            value = model.loss(X, y)

            optimizer.zero_grad()
            value.backward()
            optimizer.step()        

            total_loss += value.item()
            acc += (torch.argmax(model(X), 1) == y).sum()
            rows += X.shape[0]
        if hasattr(model, 'on_epoch_end'):
            model.on_epoch_end()
        acc = (acc / rows) * 100 
        eval = eval_model(model) * 100
        lock.acquire()
        try:
            print(f"model: {model.__class__.__name__}, epoch: {epoch}, total_loss: {total_loss}, test acc: {eval}, training acc: {acc}")
        finally:
            lock.release()
        test_accuracy.append(eval.item())
        training_accuracy.append(acc.item())

    return {
        "test_accuracy": test_accuracy,
        "training_accuracy": training_accuracy,
        "label": model.__class__.__name__
    }

if __name__ == "__main__":
    pool = ProcessPool()

    models = [
        PlainModel().to(device),
        PlainModelBatchNorm().to(device),
        PlainModelBatchNormAugmentation().to(device),
        PlainModelBatchNormAugmentationDropout().to(device),
        PlainModelDropout().to(device),
        PlainModelResidual().to(device),
        PlainModelBatchNormDynamicLr().to(device),
    ]

    for item in pool.execute(evaluate, models):
        training_accuracy, test_accuracy, name = item["training_accuracy"], item["test_accuracy"], item["label"]        
        training_plots.append(LinePlot(y=training_accuracy, title="Training accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))
        test_plots.append(LinePlot(y=test_accuracy, title="Test accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))

    plot = SimplePlot()
    plot.plot(training_plots)
    plot.save("training_accuracy.png")

    plot = SimplePlot()
    plot.plot(test_plots)
    plot.save("test_accuracy.png")
