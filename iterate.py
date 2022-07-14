from email.generator import DecodedGenerator
from torch import Tensor
import torch
from model import DecoderModel, EncoderModel
import torch.optim as optim
import torch.nn as nn

class Iterator:
    def __init__(self, encoder: EncoderModel, decoder: DecoderModel, beginning_token, end_token) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.beginning_token = beginning_token
        self.end_token = end_token

        # TODO: implement equation 5.3
        self.encoder_optimizer: optim.Adam = optim.Adam(self.encoder.parameters(), lr=0.00001)
        self.decoder_optimizer: optim.Adam = optim.Adam(self.decoder.parameters(), lr=0.00001)
        self.loss_function = nn.NLLLoss()
        self.iteration = 0
        self.accumulated_loss = 0

    def iterate(self, X: Tensor, y:Tensor):
        loss = torch.zeros(1)
        hidden = self.encoder.get_empty_hidden()
        #encoder_outputs = torch.zeros(100, self.encoder.vocab_size)

        for index in range(X.size(0)):
            output, hidden = self.encoder(X[index], hidden)
        #    encoder_outputs[index] = output[0, 0]

        decoder_input = torch.zeros(1, self.decoder.input_size)
        decoder_input[0] = self.beginning_token
        decoder_hidden = hidden
        for index in range(y.size(0)):
            # TODO: encoder input should be passed in here
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.type(torch.long), decoder_hidden.type(torch.long))
            _, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()
            expected = y[0][index]

            expected_tensor = torch.zeros((1, self.decoder.vocab_size), dtype=torch.long)
            expected_tensor[0][expected] = 1

            loss += self.loss_function(decoder_output[0], expected_tensor[0])

            if expected == self.end_token:
                break
        
        loss.backward()

        if self.iteration % 30 == 0:
            print(self.accumulated_loss)
            self.accumulated_loss = 0
        else:
            self.accumulated_loss += loss

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.iteration += 1

        return loss

