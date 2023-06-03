from dataset import get_simple_dataset, get_mnist_dataloader
from model import *
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_simple():
    layers = []
    layers.append(FfLinear(3, 1))
    layers.append(nn.Sigmoid())

    model = Model(layers)
    positive, negative = get_simple_dataset()
    for i in range(10_000):
        error = model.train(
            positive,
            negative
        )
        if i % 128 == 0:
            print(error)
    print(model.forward(positive))
    print(model.forward(negative))


def train_mnist():
    print(device)
    train_dataloader,test_loader = get_mnist_dataloader()
    layers = []
    layers.append(FfLinear(794, 256))
   # layers.append(nn.Sigmoid())
    layers.append(FfLinear(256, 128))
   # layers.append(nn.Sigmoid())
    layers.append(FfLinear(128, 256))

    model = Model(layers).to(device)

    for epoch in range(100):
        acc = torch.tensor((0), device=device, dtype=torch.float)
        batch_test = 0
        for i, (X, y) in enumerate(test_loader):
            X = X.reshape((X.shape[0], -1)).to(device)
            y = y.to(device).float()
            predicted = predict_mnist(model, X)

            acc += ((y == predicted).sum() / y.shape[0]) * 100
            batch_test += X.shape[0]        
      #      print(i)
      #      break
        print(f"({epoch}) acc == {acc / batch_test}%")

        for i, (X, y) in enumerate(train_dataloader):
            X = X.reshape((X.shape[0], -1)).to(device)
            y = torch.nn.functional.one_hot(y, 10).to(device)

            z_positive = torch.concat((
                y,
                X,
            ), dim=1)
            
            delta = torch.nn.functional.one_hot(torch.randint(size=(y.shape[0], 1), low=0, high=10, device=device).reshape(-1, 1), 10).reshape(-1, 10)
            delta -= y.reshape((-1, 10))
            
            z_negative = torch.concat((
                delta,
                X,
            ), dim=1)

            error = model.train(
                z_positive,
                z_negative
            )
            if i % 128 == 0:
                print(f"\t{error}")



def predict_mnist(model, X):
    rows = []
    """
    for _ in range(X.shape[0]):
        scores = []
        for _ in range(10):
            X = X[0].reshape((1, -1))
            y_vec = torch.nn.functional.one_hot(torch.randint(size=(1, 1), low=0, high=10, device=device), 10).reshape((-1, 10))
            z = torch.concat((
                y_vec,
                X,
            ), dim=1)
            scores.append(
                model.forward(z).item()
            )
        rows.append(scores)
    """
    output = []
    for _ in range(10):
        y_vec = torch.nn.functional.one_hot(torch.randint(size=(X.shape[0], 1), low=0, high=10, device=device), 10).reshape((-1, 10))
        z = torch.concat((
            y_vec,
            X,
        ), dim=1)
        score_class = model.forward(z).reshape((-1, 1))
        output.append(score_class)
    output_tensor_x = torch.concat(output, dim=1)
    return torch.argmax(output_tensor_x, dim=1)

if __name__ == "__main__":
#    train_simple()
    train_mnist()
