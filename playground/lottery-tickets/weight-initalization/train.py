from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
from PlainModel import PlainModel
from eval import eval_model
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_utils.utils.ProcessPool import ProcessPool
from weight_initialization import set_model_weights, ZerO_Init_on_matrix, xavier_initialization, he_initalization

EPOCHS = 100
BATCH_SIZE = 256
RANDOM_NETWORKS_TO_TRAIN = 50

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
            print(f"model: {model.name}, epoch: {epoch}, total_loss: {total_loss}, test acc: {eval}, training acc: {acc}")
        finally:
            lock.release()
        test_accuracy.append(eval.item())
        training_accuracy.append(acc.item())

    return {
        "test_accuracy": test_accuracy,
        "training_accuracy": training_accuracy,
        "label": model.name
    }

if __name__ == "__main__":
#    pool = ProcessPool(max_workers=5)

    models = []
    for i in range(RANDOM_NETWORKS_TO_TRAIN):
        models.append(PlainModel(i).to(device))
    
    zero_name = "ZerO"

#    models.append(PlainModel(zero_name).to(device))
#    set_model_weights(models[-1], ZerO_Init_on_matrix)

    special_models = [
        ("Xavier", xavier_initialization),
        ("ZeRo", ZerO_Init_on_matrix),
        ("He", he_initalization)
    ]
    training_special = []
    testing_special = []
    processed_special_models = []
    for (name, init) in special_models:
        model = PlainModel(name).to(device)
        set_model_weights(model, init)
        models.append(model)

    for item in ProcessPool(max_workers=10).execute(evaluate, models):
        training_accuracy, test_accuracy, name = item["training_accuracy"], item["test_accuracy"], item["label"]        
        if not str(name).isnumeric():
            training_special.append(LinePlot(y=training_accuracy, title="Training accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))
            testing_special.append(LinePlot(y=test_accuracy, title="Test accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))
        else:
            training_plots.append(LinePlot(y=training_accuracy, title="Training accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))
            test_plots.append(LinePlot(y=test_accuracy, title="Test accuracy", legend=name, x_text="Epochs", y_text="Accuracy", y_min=0, y_max=100))

    sorted_by_training_accuracy, sorted_by_testing_accuracy = list(zip(*sorted(
        zip(training_plots, test_plots),
        key=lambda x: sum(x[0].y)/len(x[0].y),
    )))
    training_plots = training_special + [sorted_by_training_accuracy[0], sorted_by_training_accuracy[-1]]
    test_plots = testing_special + [sorted_by_testing_accuracy[0], sorted_by_testing_accuracy[-1]]

    plot = SimplePlot()
    plot.plot(training_plots)
    plot.save("training_accuracy.png")

    plot = SimplePlot()
    plot.plot(test_plots)
    plot.save("test_accuracy.png")

    variance_count = 6
    variances = [
        sorted_by_testing_accuracy[int((len(sorted_by_testing_accuracy) - 1) * ((i) / variance_count))] for i in range(variance_count)
    ]
    print(sorted_by_testing_accuracy)
    print(variances)
    plot = SimplePlot()
    plot.plot(variances)
    plot.save("test_variance_accuracy.png")
    
    print("Okidoki -> Plotting now")
