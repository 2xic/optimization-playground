from training.model import (
    Config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Model,
    MaskOrder,
    SamplingMethod,
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
    argmax_sampling,
)
from datasets.web.web_dataset import WebDatasetSmall, WebDataset
from datasets.bytecode.bytecode_dataset import (
    BytecodeDatasetTiny,
    BytecodeDatasetBig,
    BytecodeDatasetMedium,
)
import time
import torch.multiprocessing as mp
import torch
import hashlib
from datasets.dataset import BaseDataset
from utils.web_dataloader import WebDataloader
from dotenv import load_dotenv
from torch.distributed.pipelining import pipeline
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

load_dotenv()

# assert torch.cuda.is_available()

IS_RUNNING_DISTRIBUTED = "MASTER_ADDR" in os.environ

EPOCHS = 1_00
SAMPLE_SIZE = 1
LEARNING_RATE = 3e-4
SEQUENCE_LENGTH = 32

DATASETS = [
    # WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=128),
    WebDataloader(os.environ["WEB_DATALOADER"], "medium-web", batch_size=1024)
    # WebDataloader(os.environ["WEB_DATALOADER"], "medium-512-web", batch_size=512),
]


class NamedDataset:
    def __init__(self, name, dataset: BaseDataset):
        self.name = name
        self.dataset = dataset

    @property
    def vocab_size(self):
        return self.dataset.dataset.vocab_size

    @property
    def padding_index(self):
        return self.dataset.dataset.padding_index

    @property
    def sequence_size(self):
        return self.dataset.dataset.sequence_size

    def iter(self, **args):
        return self.dataset.dataset.iter(**args, workers=0)


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
    assert dataset.sequence_size is not None
    return Config(
        dropout=0 if isinstance(dataset, XorDataset) else 0,
        dim_embeddings=4 if isinstance(dataset, XorDataset) else 728,
        num_attention_heads=2 if isinstance(dataset, XorDataset) else 8,
        num_transformer_layers=8 if isinstance(dataset, XorDataset) else 6,
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
    dataset: TransformerDataset,
    model: Model,
    optimizer_config=AdamConfig(),
):
    # TODO: move this.
    sampler = (
        temperature_sampling
        if model.config.sampling_method == SamplingMethod.TEMPERATURE
        else argmax_sampling
    )
    trainer = Trainer(
        model,
        NextTokenPrediction(
            padding_index=dataset.padding_index,
            vocab_size=dataset.vocab_size,
            sampler=sampler,
        ),
        optimizer_config.create_optimizer(model.parameters()),
    )
    return trainer


def execute(
    dataset: TransformerDataset,
    experiment_variant,
    model: Model,
    optimizer=AdamConfig(),
    options: TrainingOptions = TrainingOptions(
        epochs=EPOCHS,
        batch_size=32,
    ),
):
    epochs_accuracy = MinMaxAvgArray()
    epochs_loss = MinMaxAvgArray()
    start = time.time()

    for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {experiment_variant}"):
        trainer = create_next_token_prediction_objective(dataset, model, optimizer)
        (accuracy, loss) = trainer.train(
            dataset,
            options,
            start=start,
        )
        # assert len(accuracy) == options.epochs
        # assert len(loss) == options.epochs
        epochs_accuracy.add(accuracy)
        epochs_loss.add(loss)
        if (
            options.sampling_timeout_minutes is not None
            and (time.time() - start) // 60 > options.sampling_timeout_minutes
        ):
            print("Hit timeout")
            break
        assert len(epochs_accuracy.min_max_avg) == len(accuracy)
    # dist.barrier()
    return experiment_variant, Results(
        accuracy=epochs_accuracy,
        loss=epochs_loss,
    )


class ExperimentDistributed:
    def __init__(self, dataset: TransformerDataset):
        self.experiments = {}
        self.dataset = dataset

    def queue(
        self,
        config: Config,
        experiment_variant,
        model=Model,
        optimizer=AdamConfig(),
        training_options=TrainingOptions(
            epochs=EPOCHS, batch_size=32, training_timeout_minutes=120
        ),
    ):
        model = model(config)
        device = torch.device(f"cuda:{dist.get_rank()}")
        model = model.to(device)
        training_options.device = device

        model = FSDP(model)
        experiment_variant, results = execute(
            self.dataset, experiment_variant, model, optimizer, training_options
        )
        self.experiments[experiment_variant] = results
        return self

    def plot(self, name):
        if dist.get_rank() == 0:
            plot_accuracy_loss(self.experiments, get_output_path(self.dataset, name))


def get_experiment_instance(dataset):
    if IS_RUNNING_DISTRIBUTED:
        return ExperimentDistributed(dataset)
    else:
        return Experiment(dataset)


class Experiment:
    def __init__(self, dataset: TransformerDataset):
        self.experiments = {}
        self.dataset = dataset
        self.queue_runs = []
        self.skip_thread = False

    def queue(
        self,
        config: Config,
        experiment_variant,
        model=Model,
        optimizer=AdamConfig(),
        training_options=TrainingOptions(
            epochs=EPOCHS,
            batch_size=32,
            training_timeout_minutes=120,
        ),
    ):
        self.queue_runs.append(
            (config, experiment_variant, model, optimizer, training_options)
        )

    def plot(self, name: str):
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            results = []
            for (
                config,
                experiment_variant,
                model,
                optimizer,
                training_options,
            ) in self.queue_runs:
                model = model(config)
                args = (
                    self.dataset,
                    experiment_variant,
                    model,
                    optimizer,
                    training_options,
                )
                if self.skip_thread:
                    results.append(execute(*args))
                else:
                    results.append(
                        pool.apply_async(
                            execute,
                            args=args,
                        )
                    )
            for i in results:
                if isinstance(i, tuple):
                    experiment_variant, results = i
                else:
                    experiment_variant, results = i.get()
                self.experiments[experiment_variant] = results
            plot_accuracy_loss(self.experiments, get_output_path(self.dataset, name))


def positional_embeddings():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
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
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
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
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
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
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
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
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
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
    #    dataset = XorDataset()
    #    dataset = Datasets().get_tiny_dataset()
    dataset = WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=128)
    config = create_default_config(
        dataset,
    )
    config.dropout = 0.2
    config.dim_embeddings = 4
    config.num_transformer_layers = 2
    config.feed_forward_layer = 32
    config.num_attention_heads = 2
    config.sampling_method = SamplingMethod.TEMPERATURE
    adam_config = AdamConfig(lr=1e-3)

    options = TrainingOptions(batch_size=128, epochs=1_00, training_timeout_minutes=20)
    experiment = Experiment(dataset)
    experiment.skip_thread = True
    config.masked_order = MaskOrder.TRIU
    experiment.queue(
        config, "Fixed mask", training_options=options, optimizer=adam_config
    )
    config.masked_order = MaskOrder.TRIL
    experiment.queue(
        config, "Bad mask", training_options=options, optimizer=adam_config
    )
    config.masked_order = MaskOrder.NONE
    experiment.queue(config, "NO mask", training_options=options, optimizer=adam_config)
    experiment.plot("training_mask_fix.png")


def train():
    mixture_of_expert_model_vs_standard()
    positional_embeddings()
    transformer_layer()
    normalization_layer()
    embedding_training()
    print("Closing down ...")


if __name__ == "__main__":
    if IS_RUNNING_DISTRIBUTED:
        dist.init_process_group("nccl")
    else:
        mp.set_start_method("spawn")
    train()
    if IS_RUNNING_DISTRIBUTED:
        dist.destroy_process_group()
