import torch
import wave
import numpy as np


def softmax_mu(x_t, mu=256):
    assert torch.max(x_t) <= 1
    assert -1 <= torch.min(x_t)
    return torch.sign(x_t) * (
        torch.log(1 + mu * torch.abs(x_t)) /
        (
            torch.log(1 + mu)
        )
    )


def get_audio_wave(file):
    spf = wave.open(file, "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "int16")
    # we normalize it

    print(np.min(signal))
    print(np.max(signal))

    print(signal.shape)
    return signal

