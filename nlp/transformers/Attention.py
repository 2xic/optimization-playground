"""
Attention!!!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def scaled_dot_product(query, keys, values, d_model, return_attention=False):
    attention = nn.Softmax()(
        (
            (query @ keys.T)
            / 
            d_model
        )
    )
    values = attention @ values

    if return_attention:
        return (
            attention,
            values
        )
    return values

if __name__ == "__main__":
    seq_len = 128
    model_output = 128

    query = torch.randn((seq_len, model_output))
    keys = torch.randn((seq_len, model_output))
    values = torch.randn((seq_len, model_output))

    attention = (
        scaled_dot_product(
            query,
            keys,
            values,
            d_model=model_output
        )
    )
    fig, axes = plt.subplots(ncols=4, nrows=1)

    for index, (values, metadata) in enumerate(
        zip(
            [query, keys, values, attention],
            [
                {"title": "query"},
                {"title": "keys"},
                {"title": "values"},
                {"title": "scaled dot attention"},
            ]
        )
    ):
        axes[index].imshow(
            values.numpy()
        )
        axes[index].set_title(metadata["title"])
        axes[index].axes.set_axis_off()
    plt.savefig("attention.png")
