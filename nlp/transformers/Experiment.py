import matplotlib.pyplot as plt
from Transformer import Transformer
from EncoderTransformer import EncoderTransformer
from train_transformer import train_transformer


loss_t, acc_t = train_transformer(lambda tokens: Transformer(tokens=tokens, num_layers=2))
loss, acc = train_transformer(lambda tokens: EncoderTransformer(tokens=tokens, num_layers=4))

fig, axes = plt.subplots(nrows=2, ncols=2)

for index, (i, title) in enumerate(
    zip(
        [acc_t, acc],
        ["transformer", "encoder only"]
    )
):
    axes[0][index].plot(i)
    axes[0][index].set_title(f"Accuracy -{title}")
    axes[0][index].get_xaxis().set_visible(False)

for index, (i, title) in enumerate(
    zip(
        [loss_t, loss],
        ["transformer", "encoder only"]
    )
):
    axes[1][index].plot(i)
    axes[1][index].set_title(f"Loss -{title}")

plt.savefig('transformer_with_decoder.png')#,bbox_inches='tight')

