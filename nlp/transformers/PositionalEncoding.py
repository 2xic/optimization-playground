import numpy as np
import matplotlib.pyplot as plt
import torch

class PositionalEncoding:
    def __init__(self) -> None:
        pass

    def encode_tensor(self, batch, d_model):
        encoding = torch.zeros((batch, d_model))
        for index in range(batch):
            for i in range(d_model // 2):
                encoding[index][2 * i] = (
                    np.sin(
                        index / (10000 ** (2*i / d_model))
                    )
                )
                encoding[index][2 * i + 1] = (
                    np.cos(
                        index / (10000 ** (2*i / d_model))
                    )
                )
        return encoding
    
    def encode(self, words, d_model):
        encoding = np.zeros((
            len(words), d_model
        ))
        for index, _ in enumerate(words):
            for i in range(d_model // 2):
                encoding[index][2 * i] = (
                    np.sin(
                        index / (10000 ** (2*i / d_model))
                    )
                )
                encoding[index][2 * i + 1] = (
                    np.cos(
                        index / (10000 ** (2*i / d_model))
                    )
                )
        return encoding
    
if __name__ == "__main__":
    plt.imshow(
        PositionalEncoding().encode([0, ] * 1000, 512).T
    )
    plt.savefig('PositionalEncoding.png')


