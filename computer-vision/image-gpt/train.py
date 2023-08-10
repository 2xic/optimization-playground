"""
Train to predict next pixel from the previous batch of pixels
"""
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
#from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
# from optimization_playground_shared.nlp.Transformer import TransformerModel, Config
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.plot.Plot import Plot, Image, Figure
import random
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from collections import defaultdict
from optimization_playground_shared.distributed.TrainWrapper import MultipleGpuTrainWrapper
from torch.utils.data.distributed import DistributedSampler

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
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            vector_state[" ".join(
                vector[max(0, index - SEQUENCE_SIZE):index])] += 1
        vectors.append(vector)
    break


X_input = torch.concat(X_input, dim=0)
y_target = torch.concat(y_target, dim=0)

print(X_input.shape)
# exit(0)

class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.training_loss = []
        self.training_accuracy = []

    def _get_model_and_optimizer(self):
        config = Config(
            vocab_size=vocab.size,
            embedding_dim=512,
            transformer_layers=4,
            attention_heads=8,
            dropout=0,
            feed_forward=256,
            padding_index=vocab.vocab.PADDING_IDX,
            sequence_size=SEQUENCE_SIZE
        )
        model = GptTransformerModel(config)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def _get_dataloader(self, device):
        return get_raw_dataloader((
            X_input.clone(),
            y_target.clone()
        ),
            batch_size=256,
            shuffle=False,
            sampler=lambda dataset: DistributedSampler(dataset, shuffle=True)
        )

    def _epoch_done(self, epoch, model, loss, accuracy, device):
        self.training_loss.append(loss)
        self.training_accuracy.append(accuracy)

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
            y = model.rollout(
                seed=input_tensor,
                steps=image_size * image_size,
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
                            "Loss": self.training_loss,
                        },
                        title="Training loss",
                        x_axes_text="Epochs",
                        y_axes_text="Loss",
                    ),
                    Figure(
                        plots={
                            "Training accuracy": self.training_accuracy,
                        },
                        title="Accuracy",
                        x_axes_text="Epochs",
                        y_axes_text="accuracy",
                    ),
                ],
                name=f'debug/training_{name}.png'
            )

#    @abc.abstractclassmethod
    def _loss(self):
        return torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
