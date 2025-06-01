from training.model import (
    Config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Model,
)
from training.layers_mixture_of_experts import MoE
from utils.plot import plot_accuracy_loss, Results, MinMaxAvgArray
from utils.transformer_dataset import XorDataset
from training.trainer import Trainer
from tqdm import tqdm
import requests
from utils.dataset_tokenizer import HuggingFaceTokenizerWrapper
from utils.transformer_dataset import (
    TransformerTextDataset,
    TransformerDataset,
    BertTextDataset,
)
from optimization_playground_shared.nlp.SimpleVocab import splitter
import os
from training.objectives import NextTokenPrediction
from training.optimizer import AdamConfig, RMSpropConfig
from training.trainer import TrainingOptions
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
)
from datasets.web.web_dataset import WebDatasetSmall
from datasets.bytecode.bytecode_dataset import BytecodeDatasetTiny
import time
import torch.multiprocessing as mp
import torch

assert torch.cuda.is_available()

EPOCHS = 10
SAMPLE_SIZE = 1
LEARNING_RATE = 3e-4
SEQUENCE_LENGTH = 32


def get_text_content():
    return requests.get(
        "https://raw.githubusercontent.com/ibz/bitcoin-whitepaper-markdown/refs/heads/master/bitcoin-whitepaper.md"
    ).text


def get_text_dataset(content, percentage=0.25):
    content = content.split("\n")
    content = "\n".join(content[: int(len(content) * percentage)])
    tokenizer = HuggingFaceTokenizerWrapper(
        "example",
        vocab_size=len(list(set(splitter(content)))) * 32,
    )
    tokenizer.train_tokenizer([content])
    dataset = TransformerTextDataset.from_documents(
        tokenizer, [content], SEQUENCE_LENGTH
    )
    # Just some upper bound for speedup
    dataset._len = int(1_000 * percentage)
    return dataset


def get_masked_tokens_dataset(content, percentage=0.25):
    content = content.split("\n")
    content = "\n".join(content[: int(len(content) * percentage)])
    tokenizer = HuggingFaceTokenizerWrapper(
        "example",
        vocab_size=len(list(set(splitter(content)))) * 32,
    )
    tokenizer.train_tokenizer([content])
    dataset = BertTextDataset.from_documents(tokenizer, [content], SEQUENCE_LENGTH)
    # Just some upper bound for speedup
    dataset._len = int(1_000 * percentage)
    return dataset


class NamedDataset:
    def __init__(self, name, dataset: TransformerDataset):
        self.name = name
        self.dataset = dataset

    @property
    def vocab_size(self):
        return self.dataset.vocab_size

    @property
    def padding_index(self):
        return self.dataset.padding_index

    @property
    def sequence_size(self):
        return self.dataset.sequence_size

    def iter(self, **args):
        return self.dataset.iter(**args, workers=0)


class Datasets:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("Creating singleton instance")
            cls._instance = super().__new__(cls)
            # Expensive initialization happens here
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure initialization happens only once
        if not self._initialized:
            print("Initializing singleton")
            self._initialized = True

            text_content = get_text_content()
            self.datasets = [
                # NamedDataset("xor", XorDataset(sequence_size=3)),
                NamedDataset(
                    "satoshi_whitepaper_tiny",
                    get_text_dataset(text_content, percentage=0.05),
                ),
                NamedDataset(
                    "satoshi_whitepaper",
                    get_text_dataset(text_content, percentage=0.25),
                ),
            ]

    @property
    def value(self):
        return self.datasets


def get_output_path(dataset: NamedDataset, filename):
    dir = os.path.join(
        os.path.dirname(__file__),
        "plots",
        dataset.name,
    )
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, filename)


def create_default_config(dataset: TransformerDataset):
    assert dataset.vocab_size is not None
    assert dataset.padding_index is not None
    return Config(
        dropout=0 if isinstance(dataset, XorDataset) else 0,
        dim_embeddings=4 if isinstance(dataset, XorDataset) else 256,
        num_attention_heads=2 if isinstance(dataset, XorDataset) else 8,
        num_transformer_layers=8 if isinstance(dataset, XorDataset) else 4,
        vocab_size=dataset.vocab_size,
        sequence_length=dataset.sequence_size,
        padding_index=dataset.padding_index,
        transformer_layer=TransformerLayerType.GPT2,
        positional_embedding=PositionalEmbeddingType.NN_EMBEDDING,
    )


class ModelConfig:
    model: torch.nn.Module
    config: Config
    optimizer: AdamConfig = AdamConfig()


def create_next_token_prediction_objective(
    config: Config,
    dataset: TransformerDataset,
    model=Model,
    optimizer_config=AdamConfig(),
):
    model = model(config)
    trainer = Trainer(
        model,
        NextTokenPrediction(
            padding_index=dataset.padding_index,
            vocab_size=dataset.vocab_size,
            sampler=temperature_sampling,
        ),
        optimizer_config.create_optimizer(model.parameters()),
    )
    return trainer


def execute(
    dataset: TransformerDataset,
    config: Config,
    experiment_variant,
    model=Model,
    optimizer=AdamConfig,
):
    epochs_accuracy = MinMaxAvgArray()
    epochs_loss = MinMaxAvgArray()
    for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {experiment_variant}"):
        trainer = create_next_token_prediction_objective(
            config, dataset, model, optimizer
        )
        (accuracy, loss) = trainer.train(
            dataset,
            TrainingOptions(
                epochs=EPOCHS,
                batch_size=32,
            ),
            # progress=lambda x: x,
        )
        assert len(accuracy) == EPOCHS
        assert len(loss) == EPOCHS
        epochs_accuracy.add(accuracy)
        epochs_loss.add(loss)
        assert len(epochs_accuracy.min_max_avg) == len(accuracy)
    return experiment_variant, Results(
        accuracy=epochs_accuracy,
        loss=epochs_loss,
    )


class Experiment:
    def __init__(self, dataset: TransformerDataset):
        self.experiments = {}
        self.dataset = dataset
        self.queue_runs = []

    def queue(
        self, config: Config, experiment_variant, model=Model, optimizer=AdamConfig()
    ):
        self.queue_runs.append((config, experiment_variant, model, optimizer))

    def plot(self, name: str):
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            results = []
            for config, experiment_variant, model, optimizer in self.queue_runs:
                results.append(
                    pool.apply_async(
                        execute,
                        args=(
                            self.dataset,
                            config,
                            experiment_variant,
                            model,
                            optimizer,
                        ),
                    )
                )
            print("How much is running in parallel? Idk")
            for i in results:
                if isinstance(i, tuple):
                    experiment_variant, results = i
                else:
                    experiment_variant, results = i.get()
                self.experiments[experiment_variant] = results
            plot_accuracy_loss(self.experiments, get_output_path(self.dataset, name))


def positional_embeddings():
    for dataset in Datasets().value:
        experiment = Experiment(dataset)
        for positional_embedding in [
            PositionalEmbeddingType.NONE,
            PositionalEmbeddingType.NN_EMBEDDING,
            PositionalEmbeddingType.SINUSOIDAL,
            PositionalEmbeddingType.ROTARY_POSITION_ENCODING,
        ]:
            config = create_default_config(
                dataset,
            ).with_positional_embedding(positional_embedding)
            experiment.queue(
                config,
                positional_embedding,
            )
        experiment.plot("positional_embeddings.png")


def transformer_layer():
    for dataset in Datasets().value:
        experiment = Experiment(dataset)
        for transformer_layer in [
            TransformerLayerType.DEEPSEEK,
            TransformerLayerType.LLAMA2,
            TransformerLayerType.LLAMA3,
            TransformerLayerType.GPT2,
            TransformerLayerType.SIMPLE,
            TransformerLayerType.SIMPLE_NO_ATTENTION,
        ]:
            config = create_default_config(
                dataset,
            ).with_transformer_layer(transformer_layer)
            experiment.queue(
                config,
                transformer_layer,
            )
        experiment.plot("transformer_layer.png")


def normalization_layer():
    for dataset in Datasets().value:
        experiment = Experiment(dataset)
        for transformer_layer in [
            NormalizationLayerType.LAYER_NORM,
            NormalizationLayerType.DyT,
        ]:
            config = create_default_config(
                dataset,
            ).with_normalization_layer(transformer_layer)
            experiment.queue(
                config,
                transformer_layer,
            )
        experiment.plot("normalization_layer.png")


def mixture_of_expert_model_vs_standard():
    for dataset in Datasets().value:
        experiment = Experiment(dataset)
        for name, model in [
            ("normal", Model),
            ("mixture of experts", MoE),
        ]:
            config = create_default_config(
                dataset,
            )
            experiment.queue(config, name, model=model)
        experiment.plot("model_vs_moe.png")


def embedding_training():
    datasets = [
        NamedDataset(
            "satoshi_whitepaper",
            get_masked_tokens_dataset(get_text_content()),
        ),
        NamedDataset(
            "bytecode_dataset_small",
            BytecodeDatasetTiny(kind="masked").create_dataset(sequence_size=256)[-1],
        ),
        NamedDataset(
            "web_dataset_small",
            WebDatasetSmall(kind="masked").create_dataset(
                sequence_size=256, recreate=True
            )[-1],
        ),
    ]
    for dataset in datasets:
        print(dataset.sequence_size)
        experiment = Experiment(dataset)
        for optimizer in [AdamConfig(), RMSpropConfig()]:
            config = create_default_config(
                dataset,
            ).with_transformer_layer(TransformerLayerType.BERT)
            experiment.queue(
                config,
                "bert_" + optimizer.__class__.__name__ + f"_{dataset.name}",
                optimizer=optimizer,
            )

        experiment.plot("masked_tokens.png")


def test_pass():
    dataset = XorDataset()
    config = create_default_config(
        dataset,
    )
    trainer = create_next_token_prediction_objective(
        config,
        dataset,
    )
    (_epochs_accuracy, _epochs_loss) = trainer.train(dataset, TrainingOptions(epochs=1))
    assert len(_epochs_loss) > 0
    assert len(_epochs_accuracy) > 0


if __name__ == "__main__":
    # test_pass()
    mp.set_start_method("spawn")
    #    mixture_of_expert_model_vs_standard()
    # positional_embeddings()
    #    transformer_layer()
    #  normalization_layer()
    embedding_training()
    time.sleep(3)
