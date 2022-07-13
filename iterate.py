from torch import Tensor
import torch
from model import DecoderModel, EncoderModel

class Iterator:
    def __init__(self, encoder: EncoderModel, decoder: DecoderModel, beginning_token, end_token) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.beginning_token = beginning_token
        self.end_token = end_token

    def iterate(self, X: Tensor, y:Tensor):
        hidden = self.encoder.get_empty_hidden()
        encoder_outputs = torch.zeros(100, self.encoder.hidden_size)

        for index in range(X.size(0)):
            output, hidden = self.encoder(X[index], hidden)
            encoder_outputs[index] = output[0, 0]

        decoder_input = torch.zeros(1, 10)
        decoder_input[0] = self.beginning_token
        decoder_hidden = hidden
        for index in range(y.size(0)):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.type(torch.long), decoder_hidden.type(torch.long))
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            expected = y[0][index]
            if expected == self.end_token:
                break

