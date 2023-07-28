"""
Without pytorch lightning
"""

from dataloader import SimClrCifar100Dataloader
from model import Net, Projection, SimClrTorch
from torch.utils.data import DataLoader
import torch
from transfer_trained_simclr import test_model
from optimization_playground_shared.plot.Plot import Plot, Figure
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_model():
    base_encoder = Net().to(device)
    projection = Projection().to(device)

    model = SimClrTorch(
        base_encoder,
        projection
    )
    return model

def clone_model(base_model):
    model_copy = create_model()
    model_copy.load_state_dict(base_model.state_dict())
    return model_copy

model = create_model()

train_loader = DataLoader(SimClrCifar100Dataloader(),
                          batch_size=128,
                          shuffle=True, 
                          num_workers=8)
optimizer = torch.optim.Adam(model.parameters())

training_loss = []
training_accuracy = []
for epoch in tqdm(range(1_500)):
    sum_loss = 0
    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model._forward(batch)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    training_loss.append(sum_loss)

    if epoch % 10 == 0:
        accuracy = test_model(model=clone_model(model))
        training_accuracy.append(accuracy)

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Loss training encoder": training_loss,
                },
                title="Loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Accuracy over time with the base model": training_accuracy,
                },
                title="Accuracy",
                x_axes_text="One data point for each 10th epoch",
                y_axes_text="Accuracy",
            )
        ],
        name='results.png'
    )

