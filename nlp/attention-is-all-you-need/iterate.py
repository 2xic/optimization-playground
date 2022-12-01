from email.generator import DecodedGenerator
from torch import Tensor
import torch
from model import DecoderModel, EncoderModel
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Iterator:
    def __init__(self, encoder: EncoderModel, decoder: DecoderModel, beginning_token, end_token, device=torch.device('cpu')) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.beginning_token = beginning_token
        self.end_token = end_token

        # TODO: implement equation 5.3
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

    def iterate(self, X: Tensor, y: Tensor):

        self.combined_optimizer.zero_grad()

        loss = torch.zeros(1, device=self.device)
        encoder_hidden = self.encoder.get_empty_hidden()
        encoder_outputs = torch.zeros(
            self.encoder.input_size, self.encoder.block_size, device=self.device)
        for index in range(X.size(1)):
            input_tensor = torch.tensor([X[0][index]], device=self.device)
            encoder_output, encoder_hidden = self.encoder(
                input_tensor, encoder_hidden)
            encoder_outputs[index] = encoder_output
            self.encoder_loop_iterations += 1
            if X[0][index] == self.end_token:
                break

        decoder_input = torch.zeros(1, device=self.device)
        decoder_input[0] = self.beginning_token
        decoder_hidden = None

        predicted = []
        learning_the_hard_way = 0.10 < np.random.rand()
        for index in range(y.size(1)):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.type(torch.long),
                encoder_outputs[index],
                decoder_hidden
            )
            _, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()
            expected = y[0][index]

            expected_tensor = torch.zeros(
                (1, self.decoder.vocab_size), dtype=torch.float, device=self.device)
            expected_tensor[0][expected] = 1

            predicted.append(decoder_input.item())

        #    print((decoder_output, expected_tensor))

            loss += self.loss_function(decoder_output[0], expected_tensor[0])

            self.decoder_loop_iterations += 1

            if learning_the_hard_way:
                decoder_input = torch.tensor(expected)

            if decoder_input.item() == self.end_token:
                break

        accumulated = None
        loss.backward()

        if self.iteration % 30 == 0 and self.iteration != 0:
            print("Backwards :)")
            self.accumulated_loss = torch.zeros(1, device=self.device)
        else:
            self.accumulated_loss += loss
            accumulated = self.accumulated_loss.item()

        self.combined_optimizer.step()

        self.iteration += 1

        return loss, predicted, accumulated
