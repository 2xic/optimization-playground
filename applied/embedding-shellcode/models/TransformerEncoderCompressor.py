import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optimization_playground_shared.plot.Plot import Plot, Figure

class TransformerEncoderCompressor(nn.Module):
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
        self.embedding = nn.Embedding(
            n_token,
            self.sequence_size,
            padding_idx=0
        )
        encoder_layers = TransformerEncoderLayer(self.sequence_size, n_head)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.Z = nn.Linear(self.sequence_size, 1024)
        self.output_embedding = nn.Linear(1024, n_token)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = nn.Sigmoid()(self.transformer_encoder(src, src_mask))
        output = nn.Sigmoid()(self.Z(output))
        output = (self.output_embedding(output))
        return output
    
    def get_raw_embedding(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = nn.Sigmoid()(self.Z(output))
        return output.mean(dim=1)
    
    def forward_group(self, X, mask, real_index):
        output = self.forward(X.long(), mask)
        predicted = output.view(X.shape[0] * self.sequence_size, self.n_token)
        return predicted

    def fit(self, X, index):
        shape = X.shape[0]
        mask = torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)

        predicted = self.forward_group(
            X,
            mask,
            index
        )
        target = X.view(-1)

        print("Output / Expected")
        print(predicted.argmax(dim=1)[:10], predicted.shape)
        print(target[:10], target.shape)
        print()

        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(predicted, target.long())
        return loss
    
class ModelWrapperEncoder:
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
                name=f'loss.png'
            )

    def predict(self, dataloader, dataset):
        X = dataset.program_tensor
        y = dataset.index_tensor

        shape = X.shape[0]
        mask = torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)
        output = self.model.get_raw_embedding(X, mask)
        assert len(output.shape) == 2
        return output.detach().numpy()
