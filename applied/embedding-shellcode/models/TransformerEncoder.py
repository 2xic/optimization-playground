import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optimization_playground_shared.plot.Plot import Plot, Figure

class TransformerEncoderModel(nn.Module):
    def __init__(self, n_token, device, sequence_size: int, n_head: int = 1, n_layers: int = 6):
        super().__init__()
        self.device = device
        # I want to have a short sequence size
        # absolute width etc 
        self.sequence_size = sequence_size
        """
        Can you make this smaller ? 
        I want sequence size 32 that can be iterated over
        """
        self.n_token = n_token
        self.output_shape = 1

        self.embedding = nn.Embedding(
            n_token,
            self.sequence_size,
            padding_idx=0
        )
        encoder_layers = TransformerEncoderLayer(self.sequence_size, n_head)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.output_embedding = nn.Linear(self.sequence_size, self.output_shape)
        self.compressed_output = nn.Linear(self.output_shape * self.sequence_size, 128)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = nn.Sigmoid()(self.output_embedding(output))
        output = output.view(output.shape[0], self.output_shape * self.sequence_size)
        output = nn.Sigmoid()(self.compressed_output(output))
        return output
    
    def forward_group(self, X, mask, real_index):
        output = self.forward(X.long(), mask)

        # Make sure we don't go above the batch size

        # Step 1: Get unique indices and their corresponding counts
        unique_indices, counts = torch.unique(real_index, return_counts=True)
        # print(unique_indices)
        index = unique_indices % X.shape[0]
        select_index = real_index % X.shape[0]
        # print(index)

        real_count = torch.zeros(len(real_index)).long()
        real_count[index] = counts

        # Step 2: Use advanced indexing to group and sum the data_tensor for each index
        grouped_sum = torch.zeros(len(real_index), output.size(1), dtype=output.dtype)
        # print(grouped_sum.shape)
        grouped_sum.index_add_(0, select_index, output)
        averages = grouped_sum / real_count[:, None]

        filtered_tensor = averages[~torch.any(averages.isnan(),dim=1)]
        return filtered_tensor

    def fit(self, X, index):
        shape = X.shape[0]
        mask = torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)

        filtered_tensor = self.forward_group(
            X,
            mask,
            index
        )

        loss = self.sim_clr_loss(filtered_tensor, filtered_tensor)
        return loss

    def sim_clr_loss(self, a, b):
        self.temperature = 0.5
        batch_size = a.shape[0]
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=self.device)).float()

        representations = torch.cat([a, b], dim=0)
        S = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(S, batch_size).to(self.device)
        sim_ji = torch.diag(S, -batch_size).to(self.device)
        positives = torch.cat([sim_ij, sim_ji], dim=0).to(self.device)

        nominator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(S / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss
    
class ModelWrapper:
    def __init__(self, model, epochs=30) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self._EPOCHS = epochs

    def fit(self, X, index):
        loss = self.model.fit(X, index)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, dataloader, dataset):
        loss_over_time = []
        for epoch in range(self._EPOCHS):
            sum_loss = 0
            print(f"Epoch: {epoch}")
            for (X, index) in dataloader:
                loss = self.fit(X, index)
                sum_loss += loss.item()
                print(epoch, loss)
            loss_over_time.append(sum_loss)
            torch.save({
                "model": self.model.state_dict(),
            }, 'model.pkt')

            """
            Loss over time
            """
            plot = Plot()
            plot.plot_figures(
                figures=[
                    Figure(
                        plots={
                            "Loss": loss_over_time,
                        },
                        title="Training loss",
                        x_axes_text="Epochs",
                        y_axes_text="Loss",
                    ),
                ],
                name=f'./loss/loss_{self.model.__class__.__name__}.png'
            )

    def predict(self, dataloader, dataset):
        X = dataset.program_tensor
        y = dataset.index_tensor

        shape = X.shape[0]
        mask = torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)
        output = self.model.forward(X, mask)
        return output.detach().numpy()
