import torch
from helpers import get_audio_wave
from model import TinyWavenet


model = TinyWavenet()
data = get_audio_wave("test.wav")
data = data.reshape((1, 1, ) + data.shape)
data = torch.from_numpy(data).float()

data = data[:, :, :102]

print(data.shape)

output = model(data)

"""
1322999
1323000
"""