"""
Train to predict next pixel from the previous batch of pixels

Scaled to one large model
"""
from torch.distributed.pipeline.sync import Pipe
import os
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformerMultiGpu import get_model_from_config, Config, _verify_module
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from collections import defaultdict
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.utils.GetParameterCoumt import get_parameter_count
import time
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling
import random
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from collections import defaultdict
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.utils.GetParameterCoumt import get_parameter_count
from optimization_playground_shared.plot.Plot import Plot, Image
from tqdm import tqdm

metrics_tracker = Tracker("image_gpt_fast_overfit")

image_size = 20
train, _ = get_dataloader(
    batch_size=32,
    transforms=Compose([
        Grayscale(),
        Resize((image_size, image_size)),
        ToTensor(),
    ])
)
vocab = SimpleVocab()

SEQUENCE_SIZE = image_size * image_size
X_input = []
y_target = []
vectors = []

vector_state = defaultdict(int)

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
            y = vocab.get_tensor(
                " ".join(vector[index:index+1]), sequence_length=1)[0]
            if len(output):
                X_input.append(x)
                y_target.append(y)
            output.append(y)
            vector_state[" ".join(vector[max(0, index - SEQUENCE_SIZE):index])] += 1
        vectors.append(vector)
        # remove to not overfit
        break
    break


X_input = torch.concat(X_input, dim=0)
y_target = torch.concat(y_target, dim=0)

config = Config(
    vocab_size=vocab.size,
    embedding_dim=1024,
    transformer_layers=8,
    attention_heads=8,
    dropout=0,
    feed_forward=256,
    padding_index=vocab.vocab.PADDING_IDX,
    sequence_size=SEQUENCE_SIZE
)
model = get_model_from_config(config)
optimizer = optim.Adam(model.parameters())
print(f"Total Trainable Params: {get_parameter_count(model)}")


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

chunks = 4
_verify_module(model)
model = Pipe(model, chunks=chunks)

dataloader = get_raw_dataloader((
    X_input.clone(),
    y_target.clone()
),
    batch_size=256,
    shuffle=True,
)
"""
# Before scaling 
# One forward pass 67.44801902770996 sec

# After scaling
# 2.854787826538086                                                                             │
# One forward pass 2.869946002960205 sec
"""


def rollout(seed, steps):
    output = []
    for index in range(steps):
        next_predicted = None
        if (len(seed) - 1) < index:
            X = torch.zeros(1, config.sequence_size).reshape(
                1, -1).to("cuda:0").long().fill_(config.padding_index)
            copy = torch.tensor(output[-config.sequence_size:]).long()
            X[0, :copy.shape[0]] = copy

            X = model(X).local_value()
            next_predicted = temperature_sampling(
                X
            ).item()
            output.append(next_predicted)
        else:
            next_predicted = seed[index].item()
            output.append(next_predicted)
    return output


def track_data(epoch, loss, accuracy):
    with torch.no_grad():
        """
        Output image
        """
        def decode_image(y):
            output_tensor = torch.zeros((image_size * image_size))
            output = []
            for index, i in enumerate(y):
                i = i.item() if torch.is_tensor(i) else i
                try:
                    output.append(float(vocab.vocab.index_vocab[i]))
                except Exception as e:
                    output.append(0)
                output_tensor[index] = output[-1]
            return output_tensor.reshape((image_size, image_size))

        random_vector = vectors[random.randint(0, len(vectors) - 1)]
        input_tensor = vocab.get_tensor(" ".join(
            random_vector[:SEQUENCE_SIZE]), sequence_length=SEQUENCE_SIZE).reshape(-1)[:(image_size * image_size) // 2]
        expected_tensor = decode_image(vocab.get_tensor(" ".join(
            random_vector[:SEQUENCE_SIZE]), sequence_length=image_size*image_size).reshape(-1))
        input_image = decode_image(input_tensor)
        y = rollout(
            seed=input_tensor,
            steps=image_size * image_size,
        )
        output_image = decode_image(y)
        inference = Plot().plot_image([
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
        ], f'inference.png')

        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=loss,
                training_accuracy=accuracy,
                prediction=Prediction.image_prediction(
                    inference
                )
            )
        )


for epoch in range(1_000):
    batch_size = 0
    accuracy = 0
    sum_loss = 0
    index = 0
    update_steps = 4
    for (X, y) in tqdm(dataloader):
        # y -> is the same device as last output on the device
        X, y = X.to("cuda:0"), y.to("cuda:7")
        start = time.time()

        predicted = model(X).local_value()
        loss = torch.nn.CrossEntropyLoss()(predicted, y)
        loss.backward()

        if index % update_steps == 0:
            optimizer.step()
            model.zero_grad()

        end = time.time()

        batch_size += X.shape[0]
        accuracy += (torch.argmax(predicted, 1) == y).sum()
        sum_loss += loss.item()
        index += 1
    
    track_data(epoch, sum_loss, (
        accuracy / batch_size
    ))
