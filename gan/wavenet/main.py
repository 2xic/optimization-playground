from operator import mod
import torch
from helpers import get_audio_wave, inverse_mu, softmax_mu
from model import TinyWavenet
import torch.nn as nn

"""
TTS example
"""

# please call Stella
text_input_tensor = torch.tensor([1, 2, 3, 4, 5])

model = TinyWavenet()
adam = torch.optim.Adam(model.parameters(), lr=1e-4)

data = get_audio_wave(
    "dataset/DR-VCTK/DR-VCTK/clean_testset_wav_16k/p232_001.wav")
data = data.reshape((1, 1, ) + data.shape)

data = torch.from_numpy(data).float()[:,:, :10]
print(data.shape)
data = softmax_mu(data).float()

print(model)

for i in range(10_000):
    output = model(data)
    target = data[:, :, :output.shape[-1]]

    pred = softmax_mu(output.float())
    
#    output = nn.KLDivLoss(reduction="batchmean", log_target=True)(pred, target)

  #  print(pred.transpose(1,2).contiguous().shape)
#    print(pred.transpose(1,2).contiguous().view(-1,256))
    output = nn.CrossEntropyLoss()(
        pred.transpose(1,2).contiguous().view(-1,256),
        target.view(-1).long()
    )

    #print(pred)
    #print(target)
    #print(output)
    #print("")

    if i % 10 == 0:
        print(output)
    if i % 100 == 0:
        print(pred)
        print(target)
        print("")

    adam.zero_grad()
    output.backward()
    adam.step()
#    break
