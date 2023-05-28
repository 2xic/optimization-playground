import torch.nn as nn

def get_sentence_sim(model, a, b):
    sentence_a = (model.predict(a))
    sentence_b = (model.predict(b))
    #print(sentence_a)
    #print(sentence_b)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return (f"{a} / {b} > sim: {cos(sentence_a, sentence_b).item()}")
