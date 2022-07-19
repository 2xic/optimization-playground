from email.generator import DecodedGenerator
from torch import Tensor
import torch
from model import DecoderModel, EncoderModel
import torch.optim as optim
import torch.nn as nn
import numpy as np

class Iterator:
    def __init__(self, encoder: EncoderModel, decoder: DecoderModel, beginning_token, end_token) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.beginning_token = beginning_token
        self.end_token = end_token

        # TODO: implement equation 5.3
        self.encoder_optimizer: optim.Adam = optim.Adam(
            self.encoder.parameters(), lr=0.00001)
        self.decoder_optimizer: optim.Adam = optim.Adam(
            self.decoder.parameters(), lr=0.00001)
        self.loss_function = nn.NLLLoss()
        self.iteration = 0
        self.accumulated_loss = 0
        self.encoder_loop_iterations = 0
        self.decoder_loop_iterations = 0

    def iterate(self, X: Tensor, y: Tensor):
        loss = torch.zeros(1)
        encoder_hidden = self.encoder.get_empty_hidden()
        #encoder_outputs = torch.zeros(100, self.encoder.vocab_size)
#        encoder_output = None
        encoder_outputs = torch.zeros(
            self.encoder.input_size, self.encoder.block_size)
        for index in range(X.size(1)):
            encoder_output, encoder_hidden = self.encoder(torch.tensor([X[0][index]]), encoder_hidden)
            encoder_outputs[index] = encoder_output
            self.encoder_loop_iterations += 1

        decoder_input = torch.zeros(1)
        decoder_input[0] = self.beginning_token
        decoder_hidden = None
        predicted = []

        learning_the_hard_way = 0.5 < np.random.rand()
        for index in range(y.size(1)):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.type(torch.long),
                encoder_outputs[index],
                decoder_hidden.type(torch.long) if decoder_hidden is not None else decoder_hidden)
            _, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()
            expected = y[0][index]

            expected_tensor = torch.zeros((1, self.decoder.vocab_size), dtype=torch.long)
            expected_tensor[0][expected] = 1

            predicted.append(topi.item())

            loss += self.loss_function(decoder_output[0], expected_tensor[0])

            self.decoder_loop_iterations += 1

            if expected == self.end_token:
                break
#        print(predicted)
#        print(y.size(0))
#        print(y.shape)
        loss.backward()
        accumulated = None

        if self.iteration % 30 == 0 and self.iteration != 0:
            self.accumulated_loss = 0
        else:
            self.accumulated_loss += loss
            accumulated = self.accumulated_loss.item()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.iteration += 1

        return loss, predicted, accumulated
