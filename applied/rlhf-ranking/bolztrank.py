"""
https://icml.cc/Conferences/2009/papers/498.pdf
https://netman.aiops.org/~peidan/ANM2021/2.MachineLearningBasics/LectureCoverage/31.LearningToRank.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_creator_list.dataloader import DocumentRankDataset
from torch.utils.data import DataLoader
from optimization_playground_shared.plot.Plot import Plot, Figure
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, embeddings_size) -> None:
        super().__init__()

        self.base_layers = nn.Sequential(*[
            nn.Linear(embeddings_size, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1),
        ])

    def forward(self, item_1, item_2):
        delta = self.base_layers(item_1) - self.base_layers(item_2)
        return F.sigmoid(delta)

    def label(self, item_1, item_2):
        return (self.forward(item_1, item_2) >= torch.tensor(0.5)).long()


def quality_component(expected, predicted, Rqi):
    return torch.nn.KLDivLoss(reduction="batchmean")(expected, predicted) / Rqi

def cost_component(Rqi):
    return Rqi

def boltz_rank_optimization(expected, predicted):
    Rqi = 1  # Rank of the item
    P = 1.0  # Normalization factor
    λ = 0.5  # Trade-off parameter
    quality = λ * quality_component(expected, predicted, Rqi) / P
    cost = (1 - λ) * cost_component(Rqi)
    Oqi = quality - cost
    return Oqi

if __name__ == "__main__":

    batch_size = 32
    model = Model(embeddings_size=1536)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = DocumentRankDataset(train=True, dataset_format="binary")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DocumentRankDataset(train=False, dataset_format="binary")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    epoch_accuracy = []
    epoch_loss = []
    epoch_test_accuracy = []
    for _ in tqdm(range(5_00)):
        sum_accuracy = torch.tensor(0.0)
        sum_loss = torch.tensor(0.0)
        count = torch.tensor(0.0)
        for (x, y, label) in train_loader:
            predicted = model(x, y)
            model.zero_grad()
       #     print((predicted, label))
            loss = boltz_rank_optimization(predicted, label)   
            loss.backward()
            optimizer.step()
            
            pseudo_label = torch.argmax(model.label(x, y), dim=0)
            label = torch.argmax(label, dim=0)
            # accuracy = (pseudo_label == label).long().sum() / label.shape[0]  * 100
            sum_loss += loss
            sum_accuracy += (pseudo_label == label).long().sum()
            count += label.shape[0]
        epoch_loss.append(sum_loss.item())
#        epoch_accuracy.append(sum_accuracy / count * 100)
        epoch_accuracy.append(sum_accuracy / count * 100)

        with torch.no_grad():
            sum_accuracy = torch.tensor(0.0)
            count = torch.tensor(0.0)
            for (x, y, label) in test_loader:
                pseudo_label = torch.argmax(model.label(x, y), dim=0)
                label = torch.argmax(label, dim=0)

                sum_accuracy += (pseudo_label == label).long().sum()
                count += label.shape[0]

            epoch_test_accuracy.append(sum_accuracy / count * 100)


    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Loss": epoch_loss,
                },
                title="Training loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Training accuracy": epoch_accuracy,
                },
                title="Accuracy",
                x_axes_text="Epochs",
                y_axes_text="accuracy",
            ),
            Figure(
                plots={
                    "Testing accuracy": epoch_test_accuracy,
                },
                title="(test) Accuracy",
                x_axes_text="Epochs",
                y_axes_text="accuracy",
            ),
        ],
        name=f'training_boltzrank.png'
    )
