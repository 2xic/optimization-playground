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
    WarmupExpDecay,
)
from typing import Callable
from training.trainer import TrainingOptions, DistributedStrategy
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)
import time
import torch.multiprocessing as mp
import torch
from utils.web_dataloader import WebDataloader
from dotenv import load_dotenv
import torch.distributed as dist
from dataclasses import dataclass
import torch
import os
from utis import get_best_gpu, estimate_cuda_size, benchmark_training
from utils.checkpoints import TrainingMetadata
from utils.load_mode_from_checkpoint import (
    load_model_from_path,
    load_raw_from_path,
    load_modeL_tag,
)
from utils.checkpoints import TrainingHistory
from utils.mixture_dataloader import WebDataloaderMixture

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

DEBUG = int(os.environ.get("DEBUG", "0")) == 1
if DEBUG:
    torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

# assert torch.cuda.is_available()

IS_RUNNING_DISTRIBUTED = "MASTER_ADDR" in os.environ
DISTRIBUTED_STRATEGY = os.environ.get("PARALLEL_STRATEGY", "DDP")  # .lower()
NUM_PROCESSES = int(os.environ.get("NUM_PROCESS", 8))
USE_GRAD_SCALER = os.environ.get("USE_GRAD_SCALER", "1") == "1"
TRAINING_TIME_MINUTES = int(os.environ.get("TRAINING_TIME_MINUTES", "60"))
BATCH_SIZE = os.environ.get("BATCH_SIZE", None)

EPOCHS = int(os.environ.get("NUM_EPOCHS", 10_000))
SAMPLE_SIZE = 1
LEARNING_RATE = 3e-4

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def _batch_size(default: int) -> int:
    return int(BATCH_SIZE) if BATCH_SIZE is not None else default


TARGET_DATASET = os.environ.get("TARGET_DATASET", "small-web").lower()
NAMED_DATASETS = {
    i.name: i
    for i in [
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "satoshi-whitepaper",
            batch_size=_batch_size(256),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "smedium-web-256",
            batch_size=_batch_size(110),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-web-256-v2",
            batch_size=_batch_size(32),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "small-web-1024",
            batch_size=_batch_size(8),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "small-web",
            batch_size=_batch_size(256),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-web",
            batch_size=_batch_size(1024),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-512-web",
            batch_size=_batch_size(64),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "smoltalk-256",
            batch_size=_batch_size(110),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "self-oss-instruct-sc2-H4-256",
            batch_size=_batch_size(110),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "everyday-conversations-256",
            batch_size=_batch_size(110),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "medium-clean-web-256",
            batch_size=_batch_size(32),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "large-clean-web-256",
            batch_size=_batch_size(32),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "opcode-tokens-256",
            batch_size=_batch_size(32),
            rank=rank,
            world_size=world_size,
        ),
        WebDataloader(
            os.environ["WEB_DATALOADER"],
            "fineweb-256",
            batch_size=_batch_size(32),
            rank=rank,
            world_size=world_size,
        ),
    ]
}

DATASETS = [NAMED_DATASETS[target] for target in TARGET_DATASET.split(",")]


def get_output_path(parent_name: str, filename):
    dir = os.path.join(
        os.path.dirname(__file__),
        "plots",
        parent_name,
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
    dataset, model: Model, optimizer_config=AdamWConfig(), lr_scheduler=None
):
    sampler = (
        temperature_sampling
        if model.config.sampling_method == SamplingMethod.TEMPERATURE
        else argmax_sampling
    )
    trainer_class = GradScalerTrainer if (USE_GRAD_SCALER and not isinstance(optimizer_config, MuonConfig)) else Trainer
    if isinstance(optimizer_config, MuonConfig):
        optimizer = optimizer_config.create_optimizer_named(model.named_parameters())
    else:
        optimizer = optimizer_config.create_optimizer(model.parameters())
    if lr_scheduler is not None:
        lr_scheduler.create_scheduler(optimizer)
    trainer = trainer_class(
        model,
        NextTokenPrediction(
            padding_index=dataset.padding_index,
            vocab_size=dataset.vocab_size,
            sampler=sampler,
        ),
        optimizer,
        lr_scheduler=lr_scheduler,
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
    import gc

    epochs_accuracy = MinMaxAvgArray()
    epochs_loss = MinMaxAvgArray()
    steps_accuracy = MinMaxAvgArray()
    steps_loss = MinMaxAvgArray()
    trainer = None
    try:
        for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {experiment_variant}"):
            trainer = create_next_token_prediction_objective(
                dataset,
                model,
                options.optimizer,
                options.lr_scheduler,
            )
            (accuracy, loss, step_acc, step_ls, epoch_at_step) = trainer.train(
                dataset,
                options,
            )
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            if step_acc:
                steps_accuracy.add(step_acc)
            if step_ls:
                steps_loss.add(step_ls)
            if trainer.has_timeout(options):
                print("Hit timeout")
                break
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
    except Exception as e:
        import traceback
        print(f"Experiment '{experiment_variant}' failed during training: {e}")
        traceback.print_exc()
        raise
    finally:
        if trainer is not None:
            trainer.metrics_tracker.close()
        if not (dist.is_initialized() and options.distributed_strategy == DistributedStrategy.FSDP):
            model.cpu()
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = Results(
        accuracy=epochs_accuracy,
        loss=epochs_loss,
        step_accuracy=steps_accuracy,
        step_loss=steps_loss,
        epoch_at_step=epoch_at_step if "epoch_at_step" in dir() else [],
    )
    return experiment_variant, result


def _execute_with_lazy_model(
    dataset_name, experiment_variant, model_config, training_options
):
    result = execute(
        NAMED_DATASETS[dataset_name],
        experiment_variant,
        model_config(),
        training_options,
    )
    torch.cuda.empty_cache()
    return result


@dataclass
class LazyModelConstruction:
    config: Config
    model: Callable[..., Model] = Model

    def __call__(self):
        return self.model(self.config)


@dataclass
class PretrainedModelConstruction:
    config: Config
    model: Model

    def __call__(self):
        return self.model


class ExperimentDistributed:
    def __init__(self, dataset):
        self.experiments = {}
        self.dataset = dataset

    def queue(
        self,
        model_config: LazyModelConstruction,
        experiment_variant,
        training_options=TrainingOptions(
            epochs=EPOCHS,
            batch_size=32,
            training_timeout_minutes=TRAINING_TIME_MINUTES,
            optimizer=AdamConfig(),
        ),
    ):
        training_options.device = torch.device(f"cuda:{dist.get_rank()}")
        experiment_variant, results = execute(
            self.dataset, experiment_variant, model_config(), training_options
        )
        self.experiments[experiment_variant] = results
        return self

    def plot(self, name):
        if dist.get_rank() == 0:
            plot_accuracy_loss(
                self.experiments, get_output_path(self.dataset.name, name)
            )

    def plot_tag(self, name):
        if dist.get_rank() == 0:
            plot_accuracy_loss(self.experiments, get_output_path("tags", name))


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
        model_config: LazyModelConstruction,
        experiment_variant,
        training_options=GET_DEFAULT_TRAINING_OPTIONS(),
    ):
        self.queue_runs.append((model_config, experiment_variant, training_options))

    def _execute(self, *args):
        if self.skip_thread:
            return execute(*args)
        else:
            return self.pool.apply_async(execute, args=args)

    def _execute_lazy(
        self, dataset_name, experiment_variant, model_config, training_options
    ):
        return self.pool.apply_async(
            _execute_with_lazy_model,
            args=(dataset_name, experiment_variant, model_config, training_options),
        )

    def _execute_plots(self):
        results = []
        for (
            model_config,
            experiment_variant,
            training_options,
        ) in self.queue_runs:
            model = model_config()
            device_id = None
            while device_id is None:
                device_id = get_best_gpu(estimate_cuda_size(model))
                time.sleep(5)
            training_options.device = torch.device(f"cuda:{device_id}")
            del model
            torch.cuda.empty_cache()
            try:
                if self.skip_thread:
                    result = self._execute(
                        self.dataset,
                        experiment_variant,
                        model_config(),
                        training_options,
                    )
                else:
                    result = self._execute_lazy(
                        self.dataset.name,
                        experiment_variant,
                        model_config,
                        training_options,
                    )
                results.append((experiment_variant, result))
            except RuntimeError as e:
                print(f"Experiment '{experiment_variant}' failed: {e}")
            if not self.skip_thread:
                time.sleep(30)

        for experiment_variant, i in results:
            try:
                if isinstance(i, tuple):
                    _, result = i
                else:
                    _, result = i.get()
                self.experiments[experiment_variant] = result
            except RuntimeError as e:
                print(f"Experiment '{experiment_variant}' failed: {e}")
            finally:
                torch.cuda.empty_cache()

    def plot(self, name: str):
        self._execute_plots()
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        plot_accuracy_loss(self.experiments, get_output_path(self.dataset.name, name))

    def plot_tag(self, name):
        self._execute_plots()
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        plot_accuracy_loss(self.experiments, get_output_path("tags", name))


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
                LazyModelConstruction(config),
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
                LazyModelConstruction(config),
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
                LazyModelConstruction(config),
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
            experiment.queue(LazyModelConstruction(config), name, model=model)
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
                LazyModelConstruction(config),
                "bert_" + optimizer.__class__.__name__ + f"_{dataset.name}",
                training_options=options,
            )

        experiment.plot("masked_tokens.png")


def residual_connections():
    for dataset in DATASETS:
        experiment = get_experiment_instance(dataset)
        for name, transformer_layer in [
            ("hyper connections", TransformerLayerType.OLMO_HYPER_CONNECTIONS),
            ("residual connections", TransformerLayerType.OLMO),
            (
                "constrained hyper connection",
                TransformerLayerType.OLMO_CONSTRAINED_HYPER_CONNECTIONS,
            ),
            (
                "identity hyper connection",
                TransformerLayerType.OLMO_IDENTITY_HYPER_CONNECTIONS,
            ),
        ]:
            config = create_default_config(
                dataset,
            ).with_transformer_layer(transformer_layer)
            options = GET_DEFAULT_TRAINING_OPTIONS()
            experiment.queue(
                LazyModelConstruction(config),
                name + f"_{dataset.name}",
                training_options=options,
            )
        experiment.plot("residual_connections.png")


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
                        LazyModelConstruction(config),
                        f"{transformer_layer}_{scheduler_str}_{optimizer.__class__.__name__}_{dataset.name}",
                        training_options=TrainingOptions(
                            epochs=EPOCHS,
                            batch_size=32,
                            training_timeout_minutes=30,
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
        LazyModelConstruction(config),
        "Fixed mask",
        training_options=options,
        optimizer=adam_config,
    )
    config.masked_order = MaskOrder.TRIL
    experiment.queue(
        LazyModelConstruction(config),
        "Bad mask",
        training_options=options,
        optimizer=adam_config,
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
            #     (TransformerLayerType.GPT2, PositionalEmbeddingType.SINUSOIDAL),
        ]:
            config = (
                create_default_config(
                    dataset,
                )
                .with_transformer_layer(transformer_layer)
                .with_positional_embedding(positional_embeddings)
            )

            training_options = GET_DEFAULT_TRAINING_OPTIONS()
            training_options.enable_checkpoints = True
            # training_options.optimizer = AdamConfig(
            #    lr=3e-4 * world_size,
            #    max_grad_norm=1.0,
            # )
            # TODO: experiment with and without world_size scaling
            training_options.optimizer = AdamConfig(
                lr=3e-4,
                max_grad_norm=1.0,
            )
            training_options.lr_scheduler = WarmupExpDecay(
                warmup_steps=10_000, decay_steps=800_000, min_lr_ratio=0.01
            )
            # Train for two full days
            # training_options.training_timeout_minutes = 60 * 24 * 2
            training_options.accumulation_steps = 1
            #            if dataset.batch_size < 64:
            #                training_options.accumulation_steps = 64 // dataset.batch_size
            training_options.batch_size = dataset.batch_size
            #            training_options.batch_size = 512
            # Try to avoid gradient explosion
            # config.num_transformer_layers = 8

            # raw_dim = min(256, round(1.6 * dataset.vocab_size**0.56))
            # config.num_attention_heads = max(1, raw_dim // 8)
            # config.num_transformer_layers = 8
            # config.dim_embeddings = config.num_attention_heads * 8

            # TODO: suggestions form talking with Claude
            head_dim = 64
            raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
            config.num_attention_heads = max(1, raw_dim // head_dim)
            config.dim_embeddings = config.num_attention_heads * head_dim
            config.num_transformer_layers = 8

            # print(Model(config))
            # exit(0)

            experiment.queue(
                LazyModelConstruction(config),
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
                    LazyModelConstruction(config),
                    experiment_name,
                    training_options=training_options,
                )
            experiment.plot(
                f"{str(transformer_layer).split('.')[-1]}_embedding_sizes.png"
            )
            torch.cuda.empty_cache()


def fine_tuning():
    # 1. Take a pre-trained model
    # 2.Run on new dataset and see what happens
    # dataset = NAMED_DATASETS["smoltalk-256"]
    dataset = WebDataloaderMixture(
        [
            NAMED_DATASETS["smoltalk-256"],
            NAMED_DATASETS["everyday-conversations-256"],
            NAMED_DATASETS["self-oss-instruct-sc2-H4-256"],
        ]
    )
    experiment = get_experiment_instance(dataset)
    base_model_name = "fineweb-256"
    checkpoint_model = "checkpoints/2026-01-18/20260118_102043/step_570108"
    # for base_model_name in ["medium-web-256", "smedium-web-256", "medium-web-256-v2"]:
    if True:
        print(f"Training on {dataset.name}")
        #        base_model, config = load_best_model_from_checkpoint(base_model_name)
        base_model, config = load_model_from_path(checkpoint_model)
        training_options = GET_DEFAULT_TRAINING_OPTIONS()
        # Lower the learning rate by 10x from what we used during training.
        training_options.optimizer.lr /= 10
        training_options.enable_checkpoints = True
        # Need to keep track of this to make life easier.
        training_options.metadata = {
            "base_model_name": base_model_name,
        }

        experiment.queue(
            PretrainedModelConstruction(config, base_model),
            base_model_name,
            training_options,
        )
    experiment.plot("finetuning_from_dataset.png")
    print(base_model)


def continue_training_from_checkpoint():
    base_model_name = "fineweb-256"
    checkpoint_tag = "pre-training-fineweb256"
    # checkpoint_model = "checkpoints/2026-01-18/20260118_102043/step_570108"
    checkpoint_model = load_modeL_tag(checkpoint_tag)
    #    base_model, config = load_best_model_from_checkpoint(base_model_name)
    base_model, config = load_model_from_path(checkpoint_model)
    _, optimizer_state, stats = load_raw_from_path(checkpoint_model)
    experiment = get_experiment_instance(NAMED_DATASETS[stats["dataset"]])
    training_options = GET_DEFAULT_TRAINING_OPTIONS()
    # Lower the learning rate by 10x from what we used during training.
    # training_options.optimizer.lr /= 10
    training_options.enable_checkpoints = True
    training_options.checkpoint_tag = checkpoint_tag
    # Need to keep track of this to make life easier.
    # print(stats["metadata"]["plots"])
    history = TrainingHistory(
        stats["metadata"]["plots"]["losses"],
        stats["metadata"]["plots"]["accuracies"],
    )
    training_options.metadata = TrainingMetadata(training_options.metadata)
    training_options.metadata["continued_training_from"] = checkpoint_model
    training_options.metadata.plots = history
    # training_options.training_timeout_minutes = 1
    training_options.optimizer.optimizer_state = optimizer_state
    training_options.metadata.batches_consumed = stats["metadata"].get(
        "batches_consumed", 0
    )
    training_options.metadata.total_batch_num = stats["steps"]
    print(training_options.metadata.batches_consumed)
    # training_options.optimizer.load_state_dict(optimizer_state)

    print("queue")
    experiment.queue(
        PretrainedModelConstruction(config, base_model),
        base_model_name,
        training_options,
    )
    print("end!")
    experiment.plot_tag(f"{checkpoint_tag}.png")


def lr_sweep():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    configs = [
        (
            "warmup_exp_decay_lr1e-4",
            AdamConfig(lr=1e-4, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "warmup_exp_decay_lr3e-4",
            AdamConfig(lr=3e-4, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "warmup_exp_decay_lr1e-3",
            AdamConfig(lr=1e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "noam_lr3e-4",
            AdamConfig(lr=3e-4, max_grad_norm=1.0),
            NoamScheduler(d_model=config.dim_embeddings, warmup_steps=2000),
        ),
    ]

    for name, optimizer_config, scheduler in configs:
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=15,
            optimizer=optimizer_config,
            lr_scheduler=scheduler,
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(config),
            name,
            training_options=options,
        )
    experiment.plot("lr_sweep.png")


def lr_sweep_v2():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    configs = [
        (
            "lr5e-4",
            AdamConfig(lr=5e-4, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "lr1e-3",
            AdamConfig(lr=1e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "lr2e-3",
            AdamConfig(lr=2e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
    ]

    for name, optimizer_config, scheduler in configs:
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=30,
            optimizer=optimizer_config,
            lr_scheduler=scheduler,
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(config),
            name,
            training_options=options,
        )
    experiment.plot("lr_sweep_v2.png")


def lr_sweep_v3():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    configs = [
        (
            "lr2e-3_warmup2k",
            AdamConfig(lr=2e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "lr3e-3_warmup2k",
            AdamConfig(lr=3e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "lr3e-3_warmup4k",
            AdamConfig(lr=3e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=4000, decay_steps=50000, min_lr_ratio=0.01),
        ),
        (
            "lr5e-3_warmup4k",
            AdamConfig(lr=5e-3, max_grad_norm=1.0),
            WarmupExpDecay(warmup_steps=4000, decay_steps=50000, min_lr_ratio=0.01),
        ),
    ]

    for name, optimizer_config, scheduler in configs:
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=30,
            optimizer=optimizer_config,
            lr_scheduler=scheduler,
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(config),
            name,
            training_options=options,
        )
    experiment.plot("lr_sweep_v3.png")


def layer_scaling():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    for num_layers in [2, 4, 8, 12, 16]:
        variant_config = Config(
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            dim_embeddings=config.dim_embeddings,
            num_attention_heads=config.num_attention_heads,
            num_transformer_layers=num_layers,
            padding_index=config.padding_index,
            positional_embedding=config.positional_embedding,
            transformer_layer=config.transformer_layer,
            feed_forward_layer=config.feed_forward_layer,
            dropout=config.dropout,
        )
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=30,
            optimizer=AdamConfig(lr=3e-3, max_grad_norm=1.0),
            lr_scheduler=WarmupExpDecay(
                warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01
            ),
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(variant_config),
            f"layers_{num_layers}",
            training_options=options,
        )
    experiment.plot("layer_scaling.png")


def embedding_scaling():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    for num_heads, dim in [(2, 128), (4, 256), (8, 512), (12, 768)]:
        variant_config = Config(
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            dim_embeddings=dim,
            num_attention_heads=num_heads,
            num_transformer_layers=config.num_transformer_layers,
            padding_index=config.padding_index,
            positional_embedding=config.positional_embedding,
            transformer_layer=config.transformer_layer,
            feed_forward_layer=config.feed_forward_layer,
            dropout=config.dropout,
        )
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=30,
            optimizer=AdamConfig(lr=3e-3, max_grad_norm=1.0),
            lr_scheduler=WarmupExpDecay(
                warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01
            ),
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(variant_config),
            f"embed_{dim}",
            training_options=options,
        )
    experiment.plot("embedding_scaling.png")


def ff_scaling():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = ExperimentMultiProcess(dataset)

    config = create_default_config(dataset)
    head_dim = 64
    raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
    config.num_attention_heads = max(1, raw_dim // head_dim)
    config.dim_embeddings = config.num_attention_heads * head_dim
    config.num_transformer_layers = 8

    for ff_size in [512, 1024, 2048, 4096]:
        variant_config = Config(
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            dim_embeddings=config.dim_embeddings,
            num_attention_heads=config.num_attention_heads,
            num_transformer_layers=config.num_transformer_layers,
            padding_index=config.padding_index,
            positional_embedding=config.positional_embedding,
            transformer_layer=config.transformer_layer,
            feed_forward_layer=ff_size,
            dropout=config.dropout,
        )
        options = TrainingOptions(
            epochs=EPOCHS,
            batch_size=dataset.batch_size,
            training_timeout_minutes=30,
            optimizer=AdamConfig(lr=3e-3, max_grad_norm=1.0),
            lr_scheduler=WarmupExpDecay(
                warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.01
            ),
            record_interval_steps=100,
        )
        experiment.queue(
            LazyModelConstruction(variant_config),
            f"ff_{ff_size}",
            training_options=options,
        )
    experiment.plot("ff_scaling.png")


def long_running_training_v2():
    dataset = NAMED_DATASETS["fineweb-256"]
    experiment = get_experiment_instance(dataset)
    for transformer_layer, positional_embeddings in [
        (TransformerLayerType.LLAMA2, PositionalEmbeddingType.NONE),
    ]:
        config = (
            create_default_config(
                dataset,
            )
            .with_transformer_layer(transformer_layer)
            .with_positional_embedding(positional_embeddings)
        )

        training_options = GET_DEFAULT_TRAINING_OPTIONS()
        training_options.enable_checkpoints = True
        training_options.optimizer = AdamConfig(
            lr=2e-3,
            max_grad_norm=1.0,
        )
        training_options.lr_scheduler = WarmupExpDecay(
            warmup_steps=4_000, decay_steps=800_000, min_lr_ratio=0.01
        )
        training_options.accumulation_steps = 1
        training_options.batch_size = dataset.batch_size
        training_options.record_interval_steps = 100

        head_dim = 64
        raw_dim = min(512, round(1.6 * dataset.vocab_size**0.56))
        config.num_attention_heads = max(1, raw_dim // head_dim)
        config.dim_embeddings = config.num_attention_heads * head_dim
        config.num_transformer_layers = 4

        experiment.queue(
            LazyModelConstruction(config),
            transformer_layer,
            training_options=training_options,
        )
    experiment.plot("long_running_training_v2.png")


def train():
    residual_connections()
    # long_running_training()
    # lr_sweep()
    # lr_sweep_v2()
    # lr_sweep_v3()
    # layer_scaling()
    # embedding_scaling()
    # ff_scaling()
    # long_running_training_v2()
    # continue_training_from_checkpoint()
    # fine_tuning()
    # benchmark()
    # mixture_of_expert_model_vs_standard()
    # transformer_layer()
    #   positional_embeddings()
    #    normalization_layer()
    # embedding_training()
    # test_speedups()
    # embedding_sizes_functions()
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
