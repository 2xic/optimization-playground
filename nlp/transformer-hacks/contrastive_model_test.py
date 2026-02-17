import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import io
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from utils.web_dataloader import WebDataloader
from utils.checkpoints import StorageBoxCheckpoint, StorageBox, Stats, TrainingHistory
from utils.load_mode_from_checkpoint import load_best_model_from_checkpoint

load_dotenv()


@dataclass
class LossResult:
    loss: torch.Tensor
    correct: int


@dataclass
class ContrastiveBatch:
    anchor: torch.Tensor
    positive: torch.Tensor
    negative: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

    def to(self, device: str) -> "ContrastiveBatch":
        return ContrastiveBatch(
            anchor=self.anchor.to(device, non_blocking=True),
            positive=self.positive.to(device, non_blocking=True),
            negative=self.negative.to(device, non_blocking=True)
            if self.negative is not None
            else None,
            labels=self.labels.to(device, non_blocking=True)
            if self.labels is not None
            else None,
        )

    @property
    def batch_size(self) -> int:
        return self.anchor.size(0)


@dataclass
class EmbeddedBatch:
    anchor: torch.Tensor
    positive: torch.Tensor
    negative: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


def info_nce_loss(
    batch: EmbeddedBatch,
    temperature: float = 0.05,
    variance_weight: float = 0.1,
) -> LossResult:
    anchor_emb = F.normalize(batch.anchor, dim=1)
    pos_emb = F.normalize(batch.positive, dim=1)

    sim_matrix = torch.mm(anchor_emb, pos_emb.t()) / temperature
    batch_labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

    if batch.labels is not None:
        pos_mask = batch.labels.bool()
        if pos_mask.sum() == 0:
            return LossResult(torch.tensor(0.0, device=anchor_emb.device), 0)
        sim_matrix_filtered = sim_matrix[pos_mask]
        batch_labels_filtered = batch_labels[pos_mask]
    else:
        sim_matrix_filtered = sim_matrix
        batch_labels_filtered = batch_labels
        pos_mask = torch.ones(
            anchor_emb.size(0), dtype=torch.bool, device=anchor_emb.device
        )

    loss_a = F.cross_entropy(sim_matrix_filtered, batch_labels_filtered)
    loss_b = F.cross_entropy(sim_matrix.T[pos_mask], batch_labels_filtered)
    contrastive_loss = (loss_a + loss_b) / 2

    var_anchor = anchor_emb.var(dim=0).mean()
    var_pos = pos_emb.var(dim=0).mean()
    var_loss = -torch.log(var_anchor + 1e-4) - torch.log(var_pos + 1e-4)

    total_loss = contrastive_loss + variance_weight * var_loss
    correct = (sim_matrix_filtered.argmax(dim=1) == batch_labels_filtered).sum().item()

    return LossResult(total_loss, correct)


def triplet_loss(
    batch: EmbeddedBatch,
    margin: float = 0.2,
) -> LossResult:
    anchor = F.normalize(batch.anchor, dim=1)
    positive = F.normalize(batch.positive, dim=1)
    negative = F.normalize(batch.negative, dim=1)

    pos_dist = 1 - F.cosine_similarity(anchor, positive)
    neg_dist = 1 - F.cosine_similarity(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin).mean()
    correct = (neg_dist > pos_dist).sum().item()
    return LossResult(loss, correct)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        x = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.net(x)


class EmbeddingModel(nn.Module):
    def __init__(self, base_model: nn.Module, projection_head: ProjectionHead):
        super().__init__()
        self.base_model = base_model
        self.projection_head = projection_head

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        base_emb = self.base_model(tokens)
        return self.projection_head(base_emb)

    def forward_pair(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([tokens_a, tokens_b], dim=0)
        base_emb = self.base_model(combined)
        base_emb_a, base_emb_b = torch.chunk(base_emb, 2, dim=0)
        return self.projection_head(base_emb_a), self.projection_head(base_emb_b)


class DatasetAdapter(ABC):
    @abstractmethod
    def adapt(self, raw_batch: dict) -> ContrastiveBatch:
        pass

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        pass


class PairAdapter(DatasetAdapter):
    @property
    def columns(self) -> list[str]:
        return ["tokens_a", "tokens_b", "is_positive"]

    def adapt(self, raw_batch: dict) -> ContrastiveBatch:
        return ContrastiveBatch(
            anchor=raw_batch["tokens_a"],
            positive=raw_batch["tokens_b"],
            labels=raw_batch["is_positive"],
        )


class TripletAdapter(DatasetAdapter):
    @property
    def columns(self) -> list[str]:
        return ["ref_tokens", "pos_tokens", "neg_tokens"]

    def adapt(self, raw_batch: dict) -> ContrastiveBatch:
        return ContrastiveBatch(
            anchor=raw_batch["ref_tokens"],
            positive=raw_batch["pos_tokens"],
            negative=raw_batch["neg_tokens"],
        )


@dataclass
class ExperimentConfig:
    name: str
    dataset_name: str
    adapter: DatasetAdapter
    loss_fn: Callable[[EmbeddedBatch], LossResult]
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    projection_hidden_dim: int = 512
    projection_output_dim: int = 128

    lr_projection: float = 1e-3
    lr_base_model: float = 1e-4
    freeze_base_model: bool = False

    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    epochs: int = 1000
    timeout_minutes: int = 60
    checkpoint_interval_minutes: int = 60


@dataclass
class ExperimentResult:
    name: str
    final_loss: float
    final_accuracy: float
    runtime_seconds: float
    steps: int


class ContrastiveTrainer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.plot = TrainingHistory()
        self.checkpoint_tracker: Optional[StorageBoxCheckpoint] = None
        self.model: Optional[EmbeddingModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.dataloader: Optional[WebDataloader] = None
        self.config: Optional[ExperimentConfig] = None

    def setup(
        self, config: ExperimentConfig, base_model: nn.Module, input_dim: int = 10000
    ):
        self.config = config
        self.plot = TrainingHistory()

        run_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_tracker = StorageBoxCheckpoint(
            run_id=run_id,
            host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
            username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
            password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
        )

        projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_output_dim,
        )

        self.model = EmbeddingModel(base_model, projection_head).to(self.device)

        if config.freeze_base_model:
            for p in self.model.base_model.parameters():
                p.requires_grad = False
            self.optimizer = torch.optim.Adam(
                self.model.projection_head.parameters(), lr=config.lr_projection
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.model.projection_head.parameters(),
                        "lr": config.lr_projection,
                    },
                    {
                        "params": self.model.base_model.parameters(),
                        "lr": config.lr_base_model,
                    },
                ]
            )

        self.dataloader = WebDataloader(
            os.environ["WEB_DATALOADER"],
            dataset_name=config.dataset_name,
            columns=config.adapter.columns,
            batch_size=1024,
        )

    def _embed_batch(self, batch: ContrastiveBatch) -> EmbeddedBatch:
        emb_a, emb_b = self.model.forward_pair(batch.anchor, batch.positive)
        emb_neg = self.model(batch.negative) if batch.negative is not None else None

        return EmbeddedBatch(
            anchor=emb_a,
            positive=emb_b,
            negative=emb_neg,
            labels=batch.labels,
        )

    def _compute_loss(self, batch: ContrastiveBatch) -> LossResult:
        batch = batch.to(self.device)
        embedded = self._embed_batch(batch)
        return self.config.loss_fn(embedded, **self.config.loss_kwargs)

    def _checkpoint(self, stats: Stats):
        self.checkpoint_tracker.checkpoint_files(
            {
                "proj_head.pt": self.model.projection_head,
                "optimizer.pt": self.optimizer,
            },
            stats,
        )

    def _save_plots(self):
        os.makedirs("plots/embedding-contrastive", exist_ok=True)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(range(1, len(self.plot.losses) + 1), self.plot.losses, "b-o")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Training Loss - {self.config.name}")
        ax1.grid(True)

        ax2.plot(range(1, len(self.plot.accuracies) + 1), self.plot.accuracies, "g-o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Training Accuracy - {self.config.name}")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"plots/embedding-contrastive/{self.config.name}.png", dpi=150)
        print(f"plots/embedding-contrastive/{self.config.name}.png")
        plt.close()

    def _build_stats(
        self, avg_loss: float, accuracy: float, runtime_seconds: float, steps: int
    ) -> Stats:
        return Stats(
            loss_average=avg_loss,
            accuracy_pct=accuracy,
            runtime_seconds=runtime_seconds,
            steps=steps,
            dataset=self.config.dataset_name,
            metadata={
                "experiment_name": self.config.name,
                "base_model_name": "opcode-tokens-256",
                "config": {
                    "lr_projection": self.config.lr_projection,
                    "lr_base_model": self.config.lr_base_model,
                    "loss_kwargs": self.config.loss_kwargs,
                    "freeze_base_model": self.config.freeze_base_model,
                },
                "plots": self.plot,
            },
        )

    def train(self) -> ExperimentResult:
        print(f"\n{'=' * 60}")
        print(f"Starting experiment: {self.config.name}")
        print(f"{'=' * 60}")

        start_time = time.time()
        timeout_seconds = self.config.timeout_minutes * 60
        last_checkpoint_time = time.time()
        batch_num = 0
        final_loss = 0.0
        final_accuracy = 0.0

        threaded_loader = self.dataloader.iter(batch_size=self.config.batch_size)

        try:
            for epoch in range(self.config.epochs):
                epoch_loss = 0
                epoch_correct = 0
                epoch_samples = 0
                steps = 0

                timing = {"forward": 0, "backward": 0, "step": 0, "samples": 0}

                pbar = tqdm(
                    threaded_loader,
                    desc=f"[{self.config.name}] Epoch {epoch + 1}/{self.config.epochs}",
                )

                for index, raw_batch in enumerate(pbar):
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        pbar.close()
                        print(f"Timeout reached ({self.config.timeout_minutes} min)")
                        final_loss = epoch_loss / max(steps, 1)
                        final_accuracy = epoch_correct / max(epoch_samples, 1) * 100
                        self._checkpoint(
                            self._build_stats(
                                final_loss, final_accuracy, elapsed, batch_num
                            )
                        )
                        return ExperimentResult(
                            name=self.config.name,
                            final_loss=final_loss,
                            final_accuracy=final_accuracy,
                            runtime_seconds=elapsed,
                            steps=batch_num,
                        )

                    batch = self.config.adapter.adapt(raw_batch)

                    t0 = time.perf_counter()
                    result = self._compute_loss(batch)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    result.loss.backward()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    if (index + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()

                    timing["forward"] += t1 - t0
                    timing["backward"] += t2 - t1
                    timing["step"] += t3 - t2
                    timing["samples"] += 1

                    epoch_loss += result.loss.item()
                    epoch_correct += result.correct
                    epoch_samples += batch.batch_size
                    steps += 1
                    batch_num += 1

                    avg_loss = epoch_loss / steps
                    accuracy = (
                        epoch_correct / epoch_samples * 100 if epoch_samples > 0 else 0
                    )
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        acc=f"{accuracy:.2f}%",
                        fwd=f"{1000 * timing['forward'] / timing['samples']:.0f}ms",
                        bwd=f"{1000 * timing['backward'] / timing['samples']:.0f}ms",
                        stp=f"{1000 * timing['step'] / timing['samples']:.0f}ms",
                        time=f"{elapsed / 60:.1f}m",
                    )

                    if (
                        time.time() - last_checkpoint_time
                        > self.config.checkpoint_interval_minutes * 60
                    ):
                        self._checkpoint(
                            self._build_stats(avg_loss, accuracy, elapsed, batch_num)
                        )
                        last_checkpoint_time = time.time()

                if steps > 0:
                    final_loss = epoch_loss / steps
                    final_accuracy = epoch_correct / epoch_samples * 100
                    self.plot.record(loss=final_loss, accuracy=final_accuracy)
                    self._save_plots()

        finally:
            if threaded_loader is not None:
                threaded_loader.cleanup()
            print(f"Shutting down experiment: {self.config.name}")
            self.checkpoint_tracker._shutdown()

        return ExperimentResult(
            name=self.config.name,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            runtime_seconds=time.time() - start_time,
            steps=batch_num,
        )


def run_experiments(
    experiments: List[ExperimentConfig], device: str = "cuda"
) -> List[ExperimentResult]:
    results = []
    trainer = ContrastiveTrainer(device=device)

    for config in experiments:
        base_model, _ = load_best_model_from_checkpoint(
            target_dataset="opcode-tokens-256", max_age_days=32
        )
        trainer.setup(config, base_model)
        result = trainer.train()
        results.append(result)

        print(f"\nCompleted: {result.name}")
        print(f"  Loss: {result.final_loss:.4f}")
        print(f"  Accuracy: {result.final_accuracy:.2f}%")
        print(f"  Runtime: {result.runtime_seconds / 60:.1f} min")
        print(f"  Steps: {result.steps}")

    return results


EXPERIMENTS = [
    # Pair dataset experiments
    ExperimentConfig(
        name="baseline",
        dataset_name="contrastive_pairs-v3",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.03, "variance_weight": 0.1},
    ),
    ExperimentConfig(
        name="high_temp",
        dataset_name="contrastive_pairs-v3",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.1, "variance_weight": 0.1},
    ),
    ExperimentConfig(
        name="no_variance_reg",
        dataset_name="contrastive_pairs-v3",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.03, "variance_weight": 0.0},
    ),
    ExperimentConfig(
        name="frozen_base",
        dataset_name="contrastive_pairs-v3",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.03, "variance_weight": 0.1},
        freeze_base_model=True,
    ),
    # Triplet dataset experiments
    ExperimentConfig(
        name="triplet_baseline",
        dataset_name="etherscan-similarity-256-v2",
        adapter=TripletAdapter(),
        loss_fn=triplet_loss,
        loss_kwargs={"margin": 0.2},
    ),
    ExperimentConfig(
        name="triplet_high_margin",
        dataset_name="etherscan-similarity-256-v2",
        adapter=TripletAdapter(),
        loss_fn=triplet_loss,
        loss_kwargs={"margin": 0.5},
    ),
    ExperimentConfig(
        name="triplet_low_margin",
        dataset_name="etherscan-similarity-256-v2",
        adapter=TripletAdapter(),
        loss_fn=triplet_loss,
        loss_kwargs={"margin": 0.1},
    ),
    ExperimentConfig(
        name="triplet_frozen_base",
        dataset_name="etherscan-similarity-256-v2",
        adapter=TripletAdapter(),
        loss_fn=triplet_loss,
        loss_kwargs={"margin": 0.2},
        freeze_base_model=True,
    ),
    # Similarity ABI
    ExperimentConfig(
        name="selector_pairs_baseline",
        dataset_name="contrastive_pairs",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.03, "variance_weight": 0.1},
    ),
    ExperimentConfig(
        name="selector_pairs_high_temp",
        dataset_name="contrastive_pairs",
        adapter=PairAdapter(),
        loss_fn=info_nce_loss,
        loss_kwargs={"temperature": 0.1, "variance_weight": 0.1},
    ),
]


def train_model(epochs: int):
    for config in EXPERIMENTS:
        config.epochs = epochs
        config.timeout_minutes = 60 * 3  # 48 * 60 * 2

    results = run_experiments(EXPERIMENTS)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r.name:20s} | Loss: {r.final_loss:.4f} | Acc: {r.final_accuracy:.2f}%")


def serve_model(port):
    app = Flask(__name__)
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    base_model, _ = load_best_model_from_checkpoint(
        target_dataset="opcode-tokens-256", max_age_days=32
    )
    proj_head_state = torch.load(
        io.BytesIO(
            storage.load_bytes(
                "checkpoints/2026-01-18/20260118_195410/step_467974/proj_head.pt"
            )
        ),
        map_location=torch.device("cpu"),
    )

    projection_head = ProjectionHead(input_dim=10000, hidden_dim=512, output_dim=128)
    projection_head.load_state_dict(proj_head_state)
    model = EmbeddingModel(base_model, projection_head)
    model.eval()

    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataset_name="opcode-tokens-256",
        columns=[],
        batch_size=1024,
    )

    def embedding():
        data = request.json
        text = data["text"]
        method = data.get("method", "mean")
        normalize = data.get("normalize", False)

        with torch.no_grad():
            doc_tensors = dataloader.tokenize([text])
            embeddings = torch.concat([model(v) for v in doc_tensors], dim=0)

        if method == "mean":
            pooled = torch.mean(embeddings, dim=0)
        elif method == "max":
            pooled = torch.max(embeddings, dim=0).values
        elif method == "first":
            pooled = embeddings[0]
        elif method == "last":
            pooled = embeddings[-1]
        elif method == "weighted_decay":
            weights = torch.arange(len(embeddings), 0, -1, dtype=torch.float)
            weights = weights / weights.sum()
            pooled = (embeddings * weights.unsqueeze(1)).sum(dim=0)
        else:
            return jsonify({"error": f"Unknown method: {method}"}), 400

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=0)

        return jsonify(
            {
                "embedding": pooled.tolist(),
                "method": method,
                "normalized": normalize,
                "num_chunks": len(doc_tensors),
            }
        )

    app.route("/embedding", methods=["POST"])(embedding)
    app.run(port=port)


def main():
    parser = argparse.ArgumentParser(description="Model training and serving CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Run all experiments")
    train_parser.add_argument("--epochs", type=int, default=1000)

    serve_parser = subparsers.add_parser("serve", help="Serve the model")
    serve_parser.add_argument("--port", type=int, default=8020)

    args = parser.parse_args()

    if args.command == "train":
        train_model(epochs=args.epochs)
    elif args.command == "serve":
        serve_model(port=args.port)
    else:
        parser.print_help()

    print("The end")


if __name__ == "__main__":
    main()
