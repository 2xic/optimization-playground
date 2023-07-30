import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TransformerEncoderModel(nn.Module):
    def __init__(self, n_token, device, n_head: int = 1, n_layers: int = 6):
        super().__init__()
        self.device = device
        # I want to have a short sequence size
        # absolute width etc 
        self.SEQUENCE_SIZE = 356
        """
        Can you make this smaller ? 
        I want sequence size 32 that can be iterated over
        """
        self.n_token = n_token
        self.OUTPUT_SHAPE = 1

        self.embedding = nn.Embedding(
            n_token,
            self.SEQUENCE_SIZE,
            padding_idx=0
        )
        encoder_layers = TransformerEncoderLayer(self.SEQUENCE_SIZE, n_head)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.output_embedding = nn.Linear(self.SEQUENCE_SIZE, self.OUTPUT_SHAPE)
        self.compressed_output = nn.Linear(self.OUTPUT_SHAPE * self.SEQUENCE_SIZE, 128)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = nn.Sigmoid()(self.output_embedding(output))
        output = output.view(output.shape[0], self.OUTPUT_SHAPE * self.SEQUENCE_SIZE)
        output = nn.Sigmoid()(self.compressed_output(output))
        return output

    def fit(self, X):
        shape = X.shape[0]
        mask = torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)
        output = self.forward(X.long(), mask)
        loss = self.sim_clr_loss(output, output)
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
    def __init__(self, model) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, X):
        loss = self.model.fit(X)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
