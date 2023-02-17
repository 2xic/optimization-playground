from dataset import get_x_y
from model import EncoderModel, DecoderModelAttn
import torch
import torch.optim as optim
import torch.nn as nn
import random
from Levenshtein import distance
import matplotlib.pyplot as plt

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

X, Y = get_x_y()
for y in Y:
    y_encoded = encoder(list(y))
vocab = len(idx_2_char)


max_length = 25
model_encoder = EncoderModel(vocab)
decoder_model = DecoderModelAttn(model_encoder.hidden_size, vocab, max_length=max_length)

optimizer_encoder = torch.optim.Adam(model_encoder.parameters())
optimizer_decoder = torch.optim.Adam(decoder_model.parameters())

criterion = nn.NLLLoss()

total_runs = 1_000
update_interval = 5

loss_over_time = []
edit_distance_over_time = []
test_edit_distance = []

dataset =  list(zip(X, Y))
split = int(len(dataset) * 0.75)
train_dataset = dataset[:split]
test_dataset = dataset[split:]

def test_output():
    acc_edit_distance = []
    with torch.no_grad():
        for X, y in test_dataset:
            encoder_hidden = None
            encoder_outputs = torch.zeros(max_length, model_encoder.hidden_size)
            x_encoded = encoder(list(x))
            y_encoded = encoder(reversed(list(x)))

            for index, tensor_x_encoded in enumerate(x_encoded):
                output, hidden = model_encoder(
                    torch.tensor([tensor_x_encoded]), 
                    encoder_hidden
                )
                encoder_outputs[index] = output
            decoder_input = torch.tensor([start_token])
            hidden = None
            eval_history_decoder_output = []
            for di in range(max_length):
                decoder_output, hidden = decoder_model(
                    decoder_input,
                    hidden,
                    encoder_outputs,
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                eval_history_decoder_output.append(topi.item())
                if decoder_input.item() == end_token:
                    break
        #    print(y)
        #    print(decoder(eval_history_decoder_output))
        #    print("")
            acc_edit_distance.append(
                distance(y, decoder(eval_history_decoder_output))
            )
    return sum(acc_edit_distance) / len(acc_edit_distance)
            
for epochs in range(total_runs):
    loss = 0
    total_loss = 0
    epoch_edit_distance = []
    dataset = list(zip(X, Y))
    random.shuffle(dataset)
    for idx, (x, y) in enumerate(dataset):
        encoder_hidden = None
        encoder_outputs = torch.zeros(max_length, model_encoder.hidden_size)
        x_encoded = encoder(list(x))
        y_encoded = encoder(reversed(list(x))) + [end_token]

        for index, tensor_x_encoded in enumerate(x_encoded):
            output, hidden = model_encoder(
                torch.tensor([tensor_x_encoded]), 
                encoder_hidden
            )
            encoder_outputs[index] = output

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
        epoch_edit_distance.append(
            distance(y, decoder(history_decoder_output))
        )
    print(f"Epoch: {epochs}")
    print(f"Input: {x}")
    print(f"Output: {decoder(history_decoder_output)}")
    print(f"Expected: {y}")
    print(f"Edit distance: {distance(y, decoder(history_decoder_output))}")
    print(f"Loss: {total_loss}")
    print(epoch_edit_distance)
    print("")
    loss_over_time.append(total_loss)
    edit_distance_over_time.append(
        sum(epoch_edit_distance) / len(epoch_edit_distance)
    )
    test_edit_distance.append(
        test_output()
    )

    if epochs % 5 == 0:        
        plt.plot(edit_distance_over_time)
        plt.title("Edit distance over time")
        plt.savefig('edit_distance.png')    
        plt.clf()

        plt.plot(loss_over_time)
        plt.title("Loss over time")
        plt.savefig('loss.png')    
        plt.clf()

        plt.plot(test_edit_distance)
        plt.title("Edit distance on test dataset")
        plt.savefig('edit_distance_test.png')    
        plt.clf()
