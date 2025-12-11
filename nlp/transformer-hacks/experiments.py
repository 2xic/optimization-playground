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
from training.trainer import Trainer, GradScalerTrainer
from tqdm import tqdm
import os
from training.objectives import NextTokenPrediction
from training.optimizer import (
    AdamConfig,
    AdamWConfig,
    RMSpropConfig,
    NoamScheduler,
    MuonConfig,
    ExponentialLR,
    WarmupExpDecay,
)
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
from utis import get_best_gpu, estimate_cuda_size, benchmark_training

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

# assert torch.cuda.is_available()

IS_RUNNING_DISTRIBUTED = "MASTER_ADDR" in os.environ
DISTRIBUTED_STRATEGY = os.environ.get("PARALLEL_STRATEGY", "DDP")  # .lower()
NUM_PROCESSES = int(os.environ.get("NUM_PROCESS", 8))
USE_GRAD_SCALER = bool(int(os.environ.get("USE_GRAD_SCALER", "1")) == "1")
TRAINING_TIME_MINUTES = int(os.environ.get("TRAINING_TIME_MINUTES", "180"))
DEBUG = int(os.environ.get("DEBUG", "0")) == 1

EPOCHS = 10_000
SAMPLE_SIZE = 1
LEARNING_RATE = 3e-4

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


TARGET_DATASET = os.environ.get("TARGET_DATASET", "small-web").lower()
NAMED_DATASETS = {
    i.name: i
    for i in [
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "satoshi-whitepaper",
            batch_size=256,
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "small-web",
            batch_size=256,
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-web",
            batch_size=1024,
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-512-web",
            batch_size=64,
            rank=rank,
            world_size=world_size,
        ),
    ]
}

DATASETS = [NAMED_DATASETS[target] for target in TARGET_DATASET.split(",")]


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
    optimizer_config=AdamWConfig(),
):
    # TODO: move this.
    sampler = (
        temperature_sampling
        if model.config.sampling_method == SamplingMethod.TEMPERATURE
        else argmax_sampling
    )
    trainer_class = Trainer if USE_GRAD_SCALER else GradScalerTrainer
    trainer = trainer_class(
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
            training_timeout_minutes=TRAINING_TIME_MINUTES,
            optimizer=AdamConfig(),
        ),
    ):
        model = model(config)
        device = torch.device(f"cuda:{dist.get_rank()}")
        model = model.to(device)
        #       print(model)
        #        exit(0)
        training_options.device = device

        if DISTRIBUTED_STRATEGY == "DDP":
            model = DDP(model, device_ids=[dist.get_rank()])
            print("Using DDP for distributed training")
        else:
            model = FSDP(model, device_ids=[dist.get_rank()])
            print("Using FSDP for distributed training")
        model.config = config
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


def GET_DEFAULT_TRAINING_OPTIONS():
    return TrainingOptions(
        epochs=EPOCHS,
        batch_size=32,
        training_timeout_minutes=TRAINING_TIME_MINUTES,
        optimizer=AdamConfig(),
    )


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
        training_options=GET_DEFAULT_TRAINING_OPTIONS(),
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
            ("mixture of experts", MoE),
            ("normal", Model),
        ]:
            config = create_default_config(
                dataset,
            )
            experiment.queue(config, name, model=model)
            torch.cuda.empty_cache()
        experiment.plot("model_vs_moe.png")


def embedding_training():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
        for optimizer in [AdamConfig(), RMSpropConfig()]:
            config = create_default_config(
                dataset,
            ).with_transformer_layer(TransformerLayerType.BERT)
            options = GET_DEFAULT_TRAINING_OPTIONS()
            options.optimizer = optimizer
            experiment.queue(
                config,
                "bert_" + optimizer.__class__.__name__ + f"_{dataset.name}",
                training_options=options,
            )

        experiment.plot("masked_tokens.png")


def test_speedups():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
        for transformer_layer in [
            TransformerLayerType.DEEPSEEK,
            TransformerLayerType.LLAMA2,
            TransformerLayerType.GPT2,
        ]:
            for with_lr_scheduler in [False, True]:
                for optimizer in [AdamConfig()]:
                    config = create_default_config(
                        dataset,
                    ).with_transformer_layer(transformer_layer)
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
                            training_timeout_minutes=2,
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                            enable_checkpoints=True,
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


def benchmark():
    dataset = WebDataloader(os.environ["WEB_DATALOADER"], "medium-web", batch_size=128)
    config = create_default_config(
        dataset,
    )
    config.dropout = 0.2
    config.dim_embeddings = 4
    config.num_transformer_layers = 2
    config.feed_forward_layer = 32
    config.num_attention_heads = 2
    config.sampling_method = SamplingMethod.TEMPERATURE

    #    benchmark_training(Model(config), dataset, AdamConfig(), batch_size=1024)
    benchmark_training(
        Model(config),
        dataset,
        MuonConfig(),
        batch_size=256,
        gradient_accumulation_steps=4,
    )
    benchmark_training(Model(config), dataset, AdamConfig(), batch_size=128)
    benchmark_training(Model(config), dataset, AdamConfig(), batch_size=256)
    benchmark_training(Model(config), dataset, AdamConfig(), batch_size=512)
    benchmark_training(
        Model(config),
        dataset,
        AdamConfig(),
        batch_size=256,
        gradient_accumulation_steps=4,
    )
    benchmark_training(Model(config), dataset, RMSpropConfig(), batch_size=512)
    benchmark_training(
        Model(config),
        dataset,
        RMSpropConfig(),
        batch_size=128,
        gradient_accumulation_steps=8,
    )


def long_running_training():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
        for transformer_layer, positional_embeddings in [
            (TransformerLayerType.LLAMA2, PositionalEmbeddingType.NONE),
            (TransformerLayerType.GPT2, PositionalEmbeddingType.SINUSOIDAL),
        ]:
            config = (
                create_default_config(
                    dataset,
                )
                .with_transformer_layer(transformer_layer)
                .with_positional_embedding(positional_embeddings)
            )

            training_options = GET_DEFAULT_TRAINING_OPTIONS()
            training_options.optimizer = AdamConfig(
                lr=3e-4 * world_size,
                max_grad_norm=1.0,
            )
            training_options.lr_scheduler = WarmupExpDecay(warmup_epochs=3, gamma=0.96)
            # Train for two full days
            # training_options.training_timeout_minutes = 60 * 24 * 2
            training_options.accumulation_steps = 1
            training_options.batch_size = dataset.batch_size
            #            training_options.batch_size = 512
            # Try to avoid gradient explosion
            # config.num_transformer_layers = 8
            config.dim_embeddings = dataset.vocab_size**0.25  # 512
            print(config.dim_embeddings)

            # print(Model(config))
            # exit(0)

            experiment.queue(
                config,
                transformer_layer,
                training_options=training_options,
            )
        experiment.plot("long_running_training.png")


def embedding_sizes_functions():
    for transformer_layer, positional_embeddings in [
        (TransformerLayerType.DEEPSEEK, PositionalEmbeddingType.NONE),
        (TransformerLayerType.LLAMA2, PositionalEmbeddingType.NONE),
        (TransformerLayerType.GPT2, PositionalEmbeddingType.SINUSOIDAL),
    ]:
        for dataset in DATASETS:
            experiment = get_experiment_instance(dataset)
            # Suggestions by claude (attributions might be incorrect)
            for experiment_name, embedding_function in [
                ("Google", lambda vocab_size: vocab_size**0.25),
                (
                    "Fast.ai's rule of thumb",
                    lambda vocab_size: min(50, (vocab_size + 1) // 2),
                ),
                (
                    'The "cardinality" heuristic',
                    lambda vocab_size: min(256, round(1.6 * vocab_size**0.56)),
                ),
            ]:
                config = (
                    create_default_config(
                        dataset,
                    )
                    .with_transformer_layer(transformer_layer)
                    .with_positional_embedding(positional_embeddings)
                )
                training_options = GET_DEFAULT_TRAINING_OPTIONS()
                training_options.accumulation_steps = 1
                training_options.batch_size = dataset.batch_size
                training_options.training_timeout_minutes = 2

                raw_dim = int(embedding_function(dataset.vocab_size))
                config.num_attention_heads = max(1, raw_dim // 8)
                config.dim_embeddings = config.num_attention_heads * 8
                assert config.dim_embeddings > 0

                experiment.queue(
                    config,
                    experiment_name,
                    training_options=training_options,
                )
            experiment.plot(
                f"{str(transformer_layer).split('.')[-1]}_embedding_sizes.png"
            )
            torch.cuda.empty_cache()


def train():
    # long_running_training()
    # benchmark()
    # mixture_of_expert_model_vs_standard()
    # transformer_layer()
    #   positional_embeddings()
    #    normalization_layer()
    # embedding_training()
    # test_speedups()
    embedding_sizes_functions()
    print("Closing down ...")


if __name__ == "__main__":
    print(
        f"Running distributed {IS_RUNNING_DISTRIBUTED}, strategy={DISTRIBUTED_STRATEGY}"
    )
    if DEBUG:
        print("Debug mode enabled")
        torch.autograd.set_detect_anomaly(True)
    if IS_RUNNING_DISTRIBUTED:
        dist.init_process_group("nccl")
    else:
        mp.set_start_method("spawn")
    train()
    if IS_RUNNING_DISTRIBUTED:
        dist.destroy_process_group()
