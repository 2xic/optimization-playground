import torch

idx_tokens = {}
str_idx = {}

def add_token(token):
    idx = len(idx_tokens)
    idx_tokens[idx] = token
    str_idx[token] = idx
add_token('<PAD>')
add_token('0')
add_token('1')
add_token('2')
add_token('3')
add_token('4')
add_token('5')
add_token('6')
add_token('7')
add_token('8')
add_token('9')
add_token("+")

def encode(number_or_str, size):
    #print(number_or_str)
    encoded = []
    for i in number_or_str:
        if i not in " ":
            encoded.append(str_idx[i])
    encoded += [str_idx["<PAD>"], ] * (size- len(encoded))
    return encoded

def decoder(tensor, size):
    for i in range(tensor.shape[0]):
        decoded = ""
        for v in range(size):
            decoded += idx_tokens[int(torch.round(tensor[i][v]))]
        yield decoded
