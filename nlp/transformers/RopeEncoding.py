"""
Very naive implementation

https://huggingface.co/blog/designing-positional-encoding
https://arxiv.org/pdf/2104.09864
https://karthick.ai/blog/2024/Rotatory-Position-Embedding-%28RoPE%29/

"""
import numpy as np
import matplotlib.pyplot as plt
import torch

class ROPE:    
    def encode(self, embeddings):
        position_encodings = self.get_rotary_position_embedding(
            max_seq_len=1000,
            d_model=512,
        )
        cos_enc, sin_enc = position_encodings[..., 0::2], position_encodings[..., 1::2]
        print(cos_enc.shape)
        embeddings[..., 0::2] = embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
        embeddings[..., 1::2] = embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
        return embeddings

    def get_rotary_position_embedding(self, max_seq_len, d_model):
        # from https://karthick.ai/blog/2024/Rotatory-Position-Embedding-%28RoPE%29/
        angle_rates = 1 / torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        angles = (torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0))
        position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
        return position_encodings

    def slow_encode(self, words, d_model):
        encoding = np.zeros((
            len(words), d_model
        ))
        for index, w in enumerate(words):
            sign = -1 if index % 2 == 0 else 1
            adjusted_index = index - sign
            cos_word = w
            sin_word = words[adjusted_index] * sign
            for i in range(d_model // 2):
                encoding[index][2 * i] = (
                    (
                        np.cos(
                            index / (10000 ** (2 * i / d_model))
                        )
                        * cos_word
                    )
                    + 
                    (
                        sin_word * 
                        np.sin(
                            index / (10000 ** (2 * i / d_model))
                        )
                    )
                )
        return encoding

if __name__ == "__main__":
    out = ROPE().encode(torch.ones((1, 1000, 512)))
    print(out.shape)
    plt.imshow(out[0].T)
    plt.savefig('ROPE.png')

