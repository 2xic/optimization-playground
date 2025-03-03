import torch
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.utils.AveragePlot import AveragePlot
"""
Xor example does work ... So confused.
"""
X = torch.tensor([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1] 
])
y = torch.tensor([[0,1,1,0]]).T


config = Config(
    vocab_size=2,
    embedding_dim=8,
    transformer_layers=4,
    attention_heads=4,
    dropout=0.05,
    feed_forward=8 * 4,
    padding_index=-1,
    sequence_length=3
)
plot = AveragePlot()
for i in range(100):
    for (model, label) in [
        (GptTransformerModel(config), 'standard'),
        (GptTransformerModel(config).with_rope_encoding(), 'rope')
    ]:
        epochs = 1024
        optimi = torch.optim.Adam(model.parameters())
        accuracy_score = []
        count_full_win = 0
        for _ in range(256):
            optimi.zero_grad()
            accuracy = 0
            sum_loss = 0
            for i in range(X.shape[0]):
                output = model(X[i].reshape((1, -1)))
                last_output = output[-1, :]

                loss = torch.nn.functional.cross_entropy(
                    last_output.reshape((1, -1)),
                    y[i]
                )
                accuracy += (
                    torch.argmax(last_output) == y[i]
                )
                loss.backward()
                sum_loss += loss.item()
            optimi.step()
            print(accuracy / y.shape[0] * 100, sum_loss)
            if accuracy == y.shape[0]:
                count_full_win += 1
            accuracy_score.append(count_full_win)
        print(count_full_win)
        plot.add(label, accuracy_score)
plot.plot("rope_xor_test.png")
