from email.generator import DecodedGenerator
from torch import Tensor
import torch
from model import DecoderModel, EncoderModel
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Iterator:
    def __init__(self, 
            encoder: EncoderModel, 
            decoder: DecoderModel, 
            beginning_token, 
            end_token, 
            device=torch.device('cpu'),
            batch_update=64
        ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.beginning_token = beginning_token
        self.end_token = end_token
        self.batch_update = batch_update

        # TODO: implement equation 5.3
#        lr = 0.0001 # 3e-4 # 0.001
        lr = 0.001
        """
        self.encoder_optimizer: optim.Adam = optim.Adam(
            self.encoder.parameters(), lr=lr)
        self.decoder_optimizer: optim.Adam = optim.Adam(
            self.decoder.parameters(), lr=lr)
        """
        self.combined_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters())
        ,lr=lr)

        self.loss_function = nn.CrossEntropyLoss() # nn.NLLLoss()
        self.iteration = 0
        self.accumulated_loss = torch.zeros(1, device=device)
        self.encoder_loop_iterations = 0
        self.decoder_loop_iterations = 0
        self.device = device

    def iterate(self, X: Tensor, y: Tensor, length):
        self.combined_optimizer.zero_grad()

        loss = torch.zeros(1, device=self.device)

        input_vector = X.reshape((1, -1, ))
        encoder_output = self.encoder(input_vector, None)

        decoder_input = torch.zeros((15, 1), device=self.device)
        decoder_input[:] = self.beginning_token
        decoder_hidden = None

        predicted = []
        decoder_output = self.decoder(
            decoder_input.type(torch.long).reshape((15, 1)),
            encoder_output.type(torch.long),
            decoder_hidden
        )

        predicted_tokens = torch.argmax(decoder_output, dim=1)
        predicted = predicted_tokens.reshape((1, 15)).tolist()[0]

        loss = self.loss_function(
            decoder_output[:length], 
            y.reshape((15)) [:length]
        )
        print(loss, decoder_output, y)

        accumulated = None
        loss.backward()
        self.combined_optimizer.step()

        self.iteration += 1

        return loss, predicted, accumulated
