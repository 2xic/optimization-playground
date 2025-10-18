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
from training.trainer import Trainer
from tqdm import tqdm
import os
from training.objectives import NextTokenPrediction
from training.optimizer import AdamConfig, RMSpropConfig, NoamScheduler
from training.trainer import TrainingOptions
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)
import time
import torch.multiprocessing as mp
import torch
from utils.web_dataloader import WebDataloader
from dotenv import load_dotenv
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import os
from utis import get_best_gpu, estimate_cuda_size

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

# assert torch.cuda.is_available()

IS_RUNNING_DISTRIBUTED = "MASTER_ADDR" in os.environ
DISTRIBUTED_STRATEGY = os.environ.get("PARALLEL_STRATEGY", "DDP").lower()
NUM_PROCESSES = int(os.environ.get("NUM_PROCESS", 8))

EPOCHS = 10_000
SAMPLE_SIZE = 1
LEARNING_RATE = 3e-4

TARGET_DATASET = os.environ.get("TARGET_DATASET", "small-web").lower()
NAMED_DATASETS = {
    i.name: i
    for i in [
        WebDataloader(
            os.environ["WEB_DATALOADER"], "satoshi-whitepaper", batch_size=256
        ),
        WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=256),
        WebDataloader(os.environ["WEB_DATALOADER"], "medium-web", batch_size=1024),
        WebDataloader(os.environ["WEB_DATALOADER"], "medium-512-web", batch_size=512),
    ]
}

DATASETS = [NAMED_DATASETS[TARGET_DATASET]]


class NamedDataset:
    def __init__(self, name, dataset):
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


def create_default_config(dataset):
    assert dataset.vocab_size is not None
    assert dataset.padding_index is not None
    assert dataset.sequence_size is not None
    return Config(
        dropout=0,
        dim_embeddings=256,
        num_attention_heads=8,
        num_transformer_layers=4,
        vocab_size=dataset.vocab_size,
        sequence_length=dataset.sequence_size,
        padding_index=dataset.padding_index,
        transformer_layer=TransformerLayerType.GPT2,
        positional_embedding=PositionalEmbeddingType.NN_EMBEDDING,
    )


def create_next_token_prediction_objective(
    dataset,
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
    dataset,
    experiment_variant,
    model: Model,
    options: TrainingOptions = TrainingOptions(
        epochs=EPOCHS,
        batch_size=32,
    ),
):
    epochs_accuracy = MinMaxAvgArray()
    epochs_loss = MinMaxAvgArray()

    for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {experiment_variant}"):
        # TODO: need to copy model for sampling to be correct.
        trainer = create_next_token_prediction_objective(
            dataset, model, options.optimizer
        )
        (accuracy, loss) = trainer.train(
            dataset,
            options,
        )
        # assert len(accuracy) == options.epochs
        # assert len(loss) == options.epochs
        epochs_accuracy.add(accuracy)
        epochs_loss.add(loss)
        if trainer.has_timeout(options):
            print("Hit timeout")
            break
        assert len(epochs_accuracy.min_max_avg) == len(accuracy)
    # dist.barrier()
    return experiment_variant, Results(
        accuracy=epochs_accuracy,
        loss=epochs_loss,
    )


class ExperimentDistributed:
    def __init__(self, dataset):
        self.experiments = {}
        self.dataset = dataset

    def queue(
        self,
        config: Config,
        experiment_variant,
        model=Model,
        training_options=TrainingOptions(
            epochs=EPOCHS,
            batch_size=32,
            training_timeout_minutes=36,
            optimizer=AdamConfig(),
        ),
    ):
        model = model(config)
        device = torch.device(f"cuda:{dist.get_rank()}")
        model = model.to(device)
        training_options.device = device

        if DISTRIBUTED_STRATEGY == "DDP":
            model = DDP(model)
        else:
            model = FSDP(model)
        experiment_variant, results = execute(
            self.dataset, experiment_variant, model, training_options
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
        return ExperimentMultiProcess(dataset)


class ExperimentMultiProcess:
    def __init__(self, dataset):
        self.experiments = {}
        self.dataset = dataset
        self.queue_runs = []
        self.skip_thread = False
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(processes=NUM_PROCESSES) if not self.skip_thread else None

    def queue(
        self,
        config: Config,
        experiment_variant,
        model=Model,
        training_options=TrainingOptions(
            epochs=EPOCHS,
            batch_size=32,
            training_timeout_minutes=360,
        ),
    ):
        self.queue_runs.append((config, experiment_variant, model, training_options))

    def _execute(self, *args):
        if self.skip_thread:
            return execute(*args)
        else:
            return self.pool.apply_async(execute, args=args)

    def plot(self, name: str):
        results = []
        for (
            config,
            experiment_variant,
            model,
            training_options,
        ) in self.queue_runs:
            model = model(config)
            device_id = None
            while device_id is None:
                device_id = get_best_gpu(estimate_cuda_size(model))
                time.sleep(5)
            # Once we have a device, just set it.
            training_options.device = torch.device(f"cuda:{device_id}")
            args = (
                self.dataset,
                experiment_variant,
                model,
                training_options,
            )
            results.append(self._execute(*args))
            if not self.skip_thread:
                time.sleep(5)

        for i in results:
            print(i)
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


def test_speedups():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
        for transformer_layer in [
            TransformerLayerType.DEEPSEEK,
            TransformerLayerType.LLAMA2,
        ]:
            for with_lr_scheduler in [False, True]:
                for optimizer in [AdamConfig()]:
                    config = create_default_config(
                        dataset,
                    ).with_transformer_layer(TransformerLayerType.LLAMA2)
                    scheduler = (
                        NoamScheduler(
                            d_model=config.dim_embeddings, factor=2.0, warmup_steps=1000
                        )
                        if with_lr_scheduler
                        else None
                    )
                    scheduler_str = (
                        "with_scheduler" if with_lr_scheduler else "no_scheduler"
                    )
                    experiment.queue(
                        config,
                        f"{transformer_layer}_{scheduler_str}_{optimizer.__class__.__name__}_{dataset.name}",
                        training_options=TrainingOptions(
                            epochs=EPOCHS,
                            batch_size=32,
                            training_timeout_minutes=10,
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                        ),
                    )
            experiment.plot("debugging.png")


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
    experiment = ExperimentMultiProcess(dataset)
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
    # transformer_layer()
    # mixture_of_expert_model_vs_standard()
    # positional_embeddings()
    # normalization_layer()
    # embedding_training()
    test_speedups()
    print("Closing down ...")


if __name__ == "__main__":
    print(f"Running distributed {IS_RUNNING_DISTRIBUTED}")
    if IS_RUNNING_DISTRIBUTED:
        dist.init_process_group("nccl")
    else:
        mp.set_start_method("spawn")
    train()
    if IS_RUNNING_DISTRIBUTED:
        dist.destroy_process_group()
