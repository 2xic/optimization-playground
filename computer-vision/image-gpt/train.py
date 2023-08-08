"""
Train to predict next pixel from the previous batch of pixels
"""
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
# from optimization_playground_shared.nlp.Transformer import TransformerModel, Config
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from torchvision.utils import save_image
from optimization_playground_shared.plot.Plot import Plot, Image, Figure
import random
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize
from collections import defaultdict
#train, _ = get_dataloader(
#    batch_size=32
#)
train, _ = get_dataloader(
    batch_size=32,
    transforms=Compose([
        Grayscale(),
        Resize((28, 28)),
        ToTensor(),
    ])
)
vocab = SimpleVocab()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEQUENCE_SIZE = 728
X_input = []
y_context = []
y_target = []

vectors = []

#vector_state_seen = defaultdict(int)

for X, _ in train:
    X = X.reshape(X.shape[0], -1)
    for i in X:
        vector = ["<START>"]
        for j in i:
            vector.append(str(round(j.item(), 1)))
        output = []
        for index in range(0, len(vector) - 1):
            x = vocab.get_tensor(None, sequence_length=SEQUENCE_SIZE) if len(output) == 0 else vocab.get_tensor(
                " ".join(vector[max(0, index - SEQUENCE_SIZE):index]), sequence_length=SEQUENCE_SIZE)
            y = vocab.get_tensor(" ".join(vector[index:index+1]), sequence_length=1)[0]
            if len(output):
                X_input.append(x)
                y_target.append(y)
            output.append(y)
        vectors.append(vector)
    break

print(vocab.size)


X_input = torch.concat(X_input, dim=0).to(device)
y_target = torch.concat(y_target, dim=0).to(device)

config = Config(
    vocab_size=vocab.size,
    embedding_dim=32,
    transformer_layers=8,
    attention_heads=16,
    dropout=0,
    feed_forward=256,
    padding_index=vocab.vocab.PADDING_IDX,
    sequence_size=SEQUENCE_SIZE
)
model = GptTransformerModel(config).to(device)

training_loss = []
training_accuracy = []

dataloader = get_raw_dataloader((
    X_input,
    y_target
), batch_size=128, shuffle=True)

optimizer = optim.Adam(model.parameters())
for epoch in range(10_000):
    sum_loss = 0
    accuracy = 0
    length = 0
    for mini_x, mini_y in dataloader:
        optimizer.zero_grad()
        y_prediction = model(mini_x, mini_x)
        loss = torch.nn.CrossEntropyLoss(
            ignore_index=vocab.vocab.PADDING_IDX)(y_prediction, mini_y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

        accuracy += (torch.argmax(y_prediction, 1) == mini_y).sum()
        length += mini_y.shape[0]

    print(epoch, sum_loss)
    training_loss.append(sum_loss)
    training_accuracy.append((accuracy / length) * 100)

    with torch.no_grad():
        """
        Output image
        """
        def decode_image(y):
            output_tensor = torch.zeros((28 * 28))
            output = []
            for index, i in enumerate(y):
                i = i.item() if torch.is_tensor(i) else i
                try:
                    output.append(float(vocab.vocab.index_vocab[i]))
                except Exception as e:
                    output.append(0)
                output_tensor[index] = output[-1]
            return output_tensor.reshape((28, 28))
        
        random_vector = vectors[random.randint(0, len(vectors) - 1)]
        input_tensor = vocab.get_tensor(" ".join(random_vector[:SEQUENCE_SIZE]), sequence_length=SEQUENCE_SIZE).reshape(-1)[:512]
        expected_tensor = decode_image(vocab.get_tensor(" ".join(random_vector[:SEQUENCE_SIZE]), sequence_length=728).reshape(-1))
        input_image = decode_image(input_tensor)
        y = model.rollout(
            seed=input_tensor,
            steps=28 * 28,
            device=device,
        )
        output_image = decode_image(y)
        Plot().plot_image([
            Image(
                image=input_image,
                title='input'
            ),
            Image(
                image=output_image,
                title='output'
            ),
            Image(
                image=expected_tensor,
                title='truth'
            )
        ], f'debug/{epoch}.png')

        plot = Plot()
        name = 'image_gpt'
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Loss": training_loss,
                    },
                    title="Training loss",
                    x_axes_text="Epochs",
                    y_axes_text="Loss",
                ),
                Figure(
                    plots={
                        "Training accuracy": training_accuracy,
                    },
                    title="Accuracy",
                    x_axes_text="Epochs",
                    y_axes_text="accuracy",
                ),
            ],
            name=f'debug/training_{name}.png'
        )
