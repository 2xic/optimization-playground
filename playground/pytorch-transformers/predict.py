import torch
"""
I want the model to speak
"""
def speak(model, text, src_mask, vocab):
    output = (model(text, src_mask)[0])
    print(output.shape)
    output = torch.argmax(output, dim=1)#.shape
    print(output.shape)
    print(vocab.lookup_tokens(output.tolist()))
