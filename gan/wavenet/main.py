from operator import mod
import torch
from helpers import get_audio_wave, inverse_mu, softmax_mu, write_audio_frames
from model import TinyWavenet
import torch.nn as nn

"""
TTS example
"""
#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
#print(available_gpus)

#device = torch.device('cuda')

# please call Stella
text_input_tensor = torch.tensor([1, 2, 3, 4, 5])
model = TinyWavenet()#.to(device)
adam = torch.optim.Adam(model.parameters(), lr=1e-4)

data = get_audio_wave(
    "dataset/DR-VCTK/DR-VCTK/clean_testset_wav_16k/p232_001.wav")
data = data.reshape((1, 1, ) + data.shape)

data = torch.from_numpy(data).float()[:, :, :]
print(data.shape)
data = softmax_mu(data).float()

print(model)


def iterate_over_data():
    batch_size = 100
    for i in range(0, (data.shape[-1] // batch_size) - 2):
        j = i +1
        yield (
            data[:, :, i * batch_size: (i + 1) * batch_size],
            data[:, :, j * batch_size: (j + 1) * batch_size]
        )


for i in range(10_00):
    for v, (batch_data, next_batch_data) in enumerate(iterate_over_data()):
        output = model(batch_data)
        target = next_batch_data[:, :, :output.shape[-1]]

        pred = softmax_mu(output.float())

        loss =  ((pred - target ) ** 2).sum()

        if i % 10 == 0 and i > 0:
            print(i, loss)

        adam.zero_grad()
        loss.backward()
        adam.step()
    #    break
   # break

write_audio_frames(
    inverse_mu(data[0, 0, :]).numpy(),
    "converted_short.wav"
)
output = []
for (batch_data, next_batch_data) in iterate_over_data():
    output.append(model(batch_data)[0, 0, :])

prediction_sound = torch.concat(output, dim=0).detach()
write_audio_frames(
    prediction_sound.numpy(),
    "prediction_sound.wav"
)
