from operator import mod
import torch
from helpers import get_audio_wave, inverse_mu, softmax_mu
from model import TinyWavenet
import torch.nn as nn

"""
TTS example
"""

# please call Stella
text_input_tensor = torch.tensor([1, 2, 3, 4, ])

model = TinyWavenet()
adam = torch.optim.Adam(model.parameters(), lr=1e-4)

data = get_audio_wave(
    "dataset/DR-VCTK/DR-VCTK/clean_testset_wav_16k/p232_001.wav")
data = data.reshape((1, 1, ) + data.shape)
data = torch.from_numpy(data).float()
target = data
data = softmax_mu(data).float()

print(model)

for i in range(100):
    output = model(data)
    target = data[:, :, :output.shape[-1]]

    pred = softmax_mu(output.float())
    
    output = nn.CrossEntropyLoss()(pred, target)

    print(pred)
    print(target)
    print(output)
    print("")

    adam.zero_grad()
    output.backward()
    adam.step()
