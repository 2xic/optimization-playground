import torch.nn as nn

def get_sentence_sim(model, a, b):
    compress_if_multiple = lambda x: x.sum(dim=0).reshape(1, -1) / x.shape[0] if len(x.shape) > 1 else x
    sentence_a = compress_if_multiple(model.predict(a))
    sentence_b = compress_if_multiple(model.predict(b))
    #print(sentence_a)
    #print(sentence_b)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    return (f"{a} / {b} > sim: {cos(sentence_a, sentence_b).item()}")
