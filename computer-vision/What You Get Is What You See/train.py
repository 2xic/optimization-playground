from dataloader import LatexDataloader
from model import EncoderModel, DecoderModel, DecoderModelAttn
import torch
import torch.optim as optim
import torch.nn as nn
import random

char_2_idx = {}
idx_2_char = {}

def encoder(word):
    if type(word) == str:
        if word not in char_2_idx:
            idx = len(char_2_idx)
            char_2_idx[word] = idx
            idx_2_char[idx] = word
        return char_2_idx[word]
    else:
        encoded = []
        for i in word:
            if i not in char_2_idx:
                idx = len(char_2_idx)
                char_2_idx[i] = idx
                idx_2_char[idx] = i
            
            encoded.append(char_2_idx[i])
        return encoded

def decoder(word):
    output = []
    for i in word:
        output.append(idx_2_char.get(i, "?"))
    return "".join(output)

start_token = encoder("SOS")
end_token = encoder("END")

loader = LatexDataloader()
for (_, y) in loader:
    y_encoded = encoder(list(y))

vocab = len(idx_2_char)
print(idx_2_char)
print(vocab)
max_length = 24
model_encoder = EncoderModel()
#decoder_model = DecoderModel(model_encoder.hidden_size, 512)
decoder_model = DecoderModelAttn(model_encoder.hidden_size, vocab, max_length=max_length)

optimizer_encoder = torch.optim.Adam(model_encoder.parameters())
optimizer_decoder = torch.optim.Adam(decoder_model.parameters())

STEP_SIZE = 25
OVERFIT_SIZE = 50# len(loader)
criterion = nn.NLLLoss()
total_runs = 1_000

for epochs in range(total_runs):
    idx = random.randint(0, 60)
    update_interval = 5
    loss = 0
    total_loss = 0
    for idx in range(idx, idx + 25):
        X, y = loader[idx]
        y_encoded = encoder(list(y))
        hidden = None
        x_y = 0

        encoder_outputs = torch.zeros(max_length, model_encoder.hidden_size)
        for i in range(0, X.shape[1], STEP_SIZE):
            for j in range(0, X.shape[2], STEP_SIZE):
                image_encoder = X[:, i:i+STEP_SIZE, j:j+STEP_SIZE]
                input_image = torch.zeros((4, STEP_SIZE, STEP_SIZE))
                input_image[:, :image_encoder.shape[1], :image_encoder.shape[2]] = image_encoder
                input_image = input_image.reshape((1, ) + input_image.shape)
                output, hidden = model_encoder(input_image.float(), hidden)
            encoder_outputs[x_y] = output
            x_y += 1

        history_decoder_output = []
        force_train = random.randint(0, total_runs) > (epochs / total_runs) * total_runs
        decoder_input = torch.tensor([start_token])
        hidden = None
        if force_train:
            for di in range(len(y_encoded)):
                decoder_output, hidden = decoder_model(
                    decoder_input,
                    hidden,
                    encoder_outputs,
                )
                _, topi = decoder_output.topk(1)
                loss += criterion(decoder_output, torch.tensor([y_encoded[di]]))
                decoder_input = torch.tensor([y_encoded[di]])
                history_decoder_output.append(topi.item())
                if decoder_input.item() == end_token:
                    break
        else:
            for di in range(len(y_encoded)):
                decoder_output, hidden = decoder_model(
                    decoder_input,
                    hidden,
                    encoder_outputs,
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, torch.tensor([y_encoded[di]]))
                history_decoder_output.append(topi.item())
                if decoder_input.item() == end_token:
                    break
        if idx  > 0 and idx % update_interval == 0:
            loss /= update_interval
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            (loss).backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            total_loss += loss.item()
            loss = 0

    print(f"Output: {decoder(history_decoder_output)}")
    print(f"Expected: {y}")
    print(f"Loss: {total_loss}")
    print("")
