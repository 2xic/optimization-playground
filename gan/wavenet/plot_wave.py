import matplotlib.pyplot as plt
from helpers import get_audio_wave

signal = get_audio_wave("test.wav")

plt.figure(1)
plt.title("Signal Wave...")
plt.plot(signal)
plt.show()
