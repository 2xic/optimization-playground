import torch
import wave
import numpy as np

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm


def softmax_mu(x_t, mu=torch.tensor([256])):
    print(x_t)
#    assert torch.max(x_t) <= 1
#    assert -1 <= torch.min(x_t)
    out = torch.sign(x_t) * (
        torch.log(1 + mu * torch.abs(x_t)) /
        (
            torch.log(1 + mu)
        )
    )
#    out = torch.bucketize(out, 2 * torch.arange(mu.item()) / mu - 1) - 1
    out = out#.long()
    return out


def inverse_mu(x_t, mu=torch.tensor([256])):
    x_t = x_t.float()
    x_t = 2 * x_t / mu.item() - 1
    
    assert torch.max(x_t) <= 1
    assert -1 <= torch.min(x_t)

    output = torch.sgn(x_t) * (
        (1 + mu)**torch.abs(x_t)
        - 1
    ) / mu
    return output


def get_audio_wave(file):
    spf = wave.open(file, "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "int16").astype(np.float32)

    signal = np.divide(signal, 2**15)

    assert np.max(signal) <= 1
    assert -1 <= np.min(signal)

    return signal

def write_audio_frames(frames):
    frames *= 2 ** 15
    frames = frames.astype(np.int16)
    spf = wave.open("test-out.wav", "w")
    spf.setnchannels(1)
    spf.setsampwidth(2)
    spf.setframerate(22050)
    spf.setnframes(1323000)
    spf.writeframes(frames)
    spf.close()

