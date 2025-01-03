"""
Train to predict next pixel from the previous batch of pixels
"""
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformerMultiGpu import Config
import torch
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.utils.GetParameterCoumt import get_parameter_count
import time
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.utils.GetParameterCoumt import get_parameter_count
from tqdm import tqdm
from kmeans_color_clustering import quantize, get_centroids_file
from optimization_playground_shared.plot.Plot import Plot, Image
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from gpt_model import GPT2
from tqdm import tqdm
import torch.nn.functional as F
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader as cifar_dataloader
from optimization_playground_shared.dataloaders.Mnist import get_dataloader as mnist_dataloader
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("image_gpt_fast_overfit")
dataset = "mnist"
get_dataloader = mnist_dataloader if dataset == "mnist" else cifar_dataloader
image_size = 20
batch_size = 32
train, _ = get_dataloader(
    batch_size=batch_size,
    transforms=Compose([
        Grayscale(),
        Resize((image_size, image_size)),
        ToTensor(),
    ])
)
vocab = SimpleVocab()

SEQUENCE_SIZE = image_size * image_size

config = Config(
    vocab_size=16,
    embedding_dim=512,
    transformer_layers=2,
    attention_heads=8,
    dropout=0,
    feed_forward=256,
    padding_index=vocab.vocab.PADDING_IDX,
    sequence_size=SEQUENCE_SIZE
)
#model = GptTransformerModel(config).to(device)
model = GPT2(config).to(device)
optimizer = optim.Adam(model.parameters())
print(f"Total Trainable Params: {get_parameter_count(model)}")


centroids = get_centroids_file()

def to_sequence(x):
    x = x.view(x.shape[0], -1)
    x = x.transpose(0, 1).contiguous()
    return x


def rollout_transformer(model, seed, steps, device):
    output = seed.unsqueeze(-1).repeat_interleave(1, dim=-1)
    pad = torch.zeros(1, 1, dtype=torch.long).to( device)
    with torch.no_grad():
        for _ in range(steps):
            # print(epocj)
            logits = model.forward(torch.cat((output, pad), dim=0))
            # print(logits.shape)
            logits = logits[-1:, :, :].squeeze(0)
            prob = F.softmax(logits, dim=-1)
            # print(prob.shape)
            pred = torch.multinomial(prob, num_samples=1).transpose(1, 0)
            output = torch.cat((output, pred), dim=0)
    return output

for epoch in range(1_000):
    batch_size = 0
    accuracy = 0
    sum_loss = 0
    index = 0
    update_steps = 4
    for (X, _) in tqdm(train):
        X = quantize(X, centroids)
        X = X.to(device)
        X = to_sequence(X)
        start = time.time()

        predicted = model.forward(X)
        loss = torch.nn.CrossEntropyLoss()(
            predicted.view(-1, predicted.shape[-1]), 
            X.view(-1)
        )
        loss.backward()

        if index % update_steps == 0:
            optimizer.step()
            model.zero_grad()

        end = time.time()

        batch_size += X.shape[0]
        sum_loss += loss.item()
        index += 1

    print(f"sum loss : {sum_loss}")
    with torch.no_grad():
        """
        Output image
        """
        def decode_image(y):
            return centroids[y.to('cpu')]
        #print(X.shape)
        copy_image = X.transpose(1, 0)
        copy_image = copy_image[random.randint(0,  copy_image.shape[0] -1) - 1]
        input_tensor = torch.zeros((200), device=device).long()
        input_tensor[:200] = copy_image[:200]
        
        #input_image = decode_image(input_tensor)
        y = rollout_transformer(
            model=model,
            seed=input_tensor,
            steps=200,
            device=device,
        )
       # print("out ", y.shape)
       # print("aaa", decode_image(input_tensor).shape)
        output_image = decode_image(y)
        inference = Plot().plot_image([
            Image(
                image=torch.concat( (decode_image(input_tensor), torch.zeros((200, 1))), dim=0 ).to('cpu').detach().reshape((1, image_size, image_size)),
                title='input'
            ),
            Image(
                image=output_image.to('cpu').detach().reshape((1, image_size, image_size)),
                title='output'
            ),
            Image(
                image=decode_image(copy_image).to('cpu').detach().reshape((1, image_size, image_size)),
                title='truth'
            )
        ], f'inference.png')

        metrics_tracker._log(
            Metrics(
                epoch=epoch,
                loss=sum_loss,
                training_accuracy=None,
                prediction=Prediction.image_prediction(
                    inference
                )
            )
        )
