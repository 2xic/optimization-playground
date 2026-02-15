import torch.nn.functional as F
import torch
from training.model import Model
import torch
import matplotlib.pyplot as plt
import time
import torch
from utils.web_dataloader import WebDataloader, FloatColumn
import os
from dotenv import load_dotenv
from utils.checkpoints import StorageBoxCheckpoint, Stats, TrainingHistory
from datetime import datetime
from tqdm import tqdm
from utils.load_mode_from_checkpoint import (
    load_best_model_from_checkpoint,
    load_model_from_path,
    load_raw_from_path,
)
import argparse
import torch.nn as nn
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify
from utils.checkpoints import StorageBox
import io


class Objective(ABC):
    dataloader: WebDataloader

    @abstractmethod
    def forward(self, dataset):
        pass

    @abstractmethod
    def checkpoint_files(self):
        pass


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        # Learned attention pooling
        self.attention = nn.Linear(input_dim, 1)

        # Projection
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask=None):
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=1)
        x = (x * weights.unsqueeze(-1)).sum(dim=1)

        return self.net(x)


class InfoObjectiveDataset:
    def __init__(self, base_model, device):
        self.base_model = base_model
        self.device = device
        self.dataloader = WebDataloader(
            os.environ["WEB_DATALOADER"],
            # dataset_name="contract-similarity-v2",
            dataset_name="contrastive_pairs-v3",
            # "etherscan-similarity-256-v2",
            # columns=["ref_tokens", "pos_tokens", "neg_tokens"],
            columns=[
                "tokens_a",
                "tokens_b",
                "is_positive",
            ],  # , FloatColumn("similarity")],
            batch_size=1024,
        )
        # self.base_model.eval()
        # for p in self.base_model.parameters():
        #    p.requires_grad = False
        self.proj_head = ProjectionHead(
            input_dim=10000, hidden_dim=512, output_dim=128
        ).to(self.device)
        self.base_model = self.base_model.to(self.device)
        self.proj_head = self.proj_head.to(self.device)
        # self.optimizer = torch.optim.Adam(self.proj_head.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.proj_head.parameters(), "lr": 1e-3},
                {"params": self.base_model.parameters(), "lr": 1e-4},
            ]
        )

    @classmethod
    def load_from_storage(self, base_model, proj_head_state):
        cls = InfoObjectiveDataset(base_model, "cpu")
        cls.proj_head.load_state_dict(proj_head_state)
        return cls

    """
    def loss(self, anchor_emb, pos_emb, temperature=0.05):
        # Normalize
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        sim_matrix = torch.mm(anchor_emb, pos_emb.t()) / temperature
        labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

        # Symmetric loss
        loss_a = F.cross_entropy(sim_matrix, labels)
        loss_b = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_a + loss_b) / 2

        correct = (sim_matrix.argmax(dim=1) == labels).sum().item()
        return loss, correct
    """

    def embed(self, tokens):
        tokens_a = tokens.to(self.device)
        base_emb_a = self.base_model(tokens_a)
        emb_a = self.proj_head(base_emb_a)
        return emb_a

    def loss(self, anchor_emb, pos_emb, is_positive, temperature=0.05):
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        sim_matrix = torch.mm(anchor_emb, pos_emb.t()) / temperature
        labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

        # Only compute loss on positive pairs
        pos_mask = is_positive.bool()
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=anchor_emb.device), 0

        loss_a = F.cross_entropy(sim_matrix[pos_mask], labels[pos_mask])
        loss_b = F.cross_entropy(sim_matrix.T[pos_mask], labels[pos_mask])
        contrastive_loss = (loss_a + loss_b) / 2

        # Variance regularization - prevents embedding collapse
        # Encourages each dimension to have high variance across the batch
        var_anchor = anchor_emb.var(dim=0).mean()
        var_pos = pos_emb.var(dim=0).mean()
        var_loss = -torch.log(var_anchor + 1e-4) - torch.log(var_pos + 1e-4)
        loss = contrastive_loss + 0.1 * var_loss

        correct = (sim_matrix[pos_mask].argmax(dim=1) == labels[pos_mask]).sum().item()
        return loss, correct

    def forward(self, batch):
        (
            tokens_a,
            tokens_b,
            is_positive,
        ) = (batch["tokens_a"], batch["tokens_b"], batch["is_positive"])

        tokens_a = tokens_a.to(self.device, non_blocking=True)
        tokens_b = tokens_b.to(self.device, non_blocking=True)

        #   with torch.no_grad():
        # base_emb_a = self.base_model(tokens_a)
        # base_emb_b = self.base_model(tokens_b)
        combined = torch.cat([tokens_a, tokens_b], dim=0)
        base_emb = self.base_model(combined)
        base_emb_a, base_emb_b = torch.chunk(base_emb, 2, dim=0)

        emb_a = self.proj_head(base_emb_a)
        emb_b = self.proj_head(base_emb_b)
        loss, accuracy = self.loss(emb_a, emb_b, is_positive, temperature=0.03)
        return loss, accuracy, tokens_a.size(0)

    def checkpoint_files(self):
        return {
            "proj_head.pt": self.proj_head,
            "optimizer.pt": self.optimizer,
        }


load_dotenv()


def random_crop(tokens, length=256):
    """Random offset crop, padded/truncated to fixed length"""
    seq_len = tokens.size(1)

    if seq_len <= length:
        raise Exception(f"This should not happen {length}")

    # Random offset
    max_offset = seq_len - length
    offset = torch.randint(0, max_offset + 1, (1,)).item()
    return tokens[:, offset : offset + length]


class EmbeddingTrainer:
    def __init__(
        self,
        plot: TrainingHistory = TrainingHistory(),
        margin=0.2,
        device="cuda",
    ):
        self.device = device
        self.margin = margin
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoints_tracker = StorageBoxCheckpoint(
            run_id=run_id,
            host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
            username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
            password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
        )
        self.last_checkpoint = time.time()
        self.plot = plot

    def embed(self, x):
        """Mean pool hidden states → normalize"""
        hidden = self.model.get_hidden_states(x)  # (batch, seq, dim)
        emb = hidden.mean(dim=1)  # (batch, dim)
        return F.normalize(emb, p=2, dim=-1)

    def triplet_loss(self, anchor, positive, negative):
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        return F.relu(pos_dist - neg_dist + self.margin).mean()

    # InfoNCE loss with in-batch negatives.
    def in_batch_contrastive_loss(self, anchor_emb, pos_emb, temperature=0.05):
        sim_matrix = torch.mm(anchor_emb, pos_emb.t()) / temperature
        labels = torch.arange(anchor_emb.size(0), device=self.device)
        loss = F.cross_entropy(sim_matrix, labels)
        correct = (sim_matrix.argmax(dim=1) == labels).sum().item()
        return loss, correct

    def train(self, objective: Objective, epochs=3, timeout_minutes=60):
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        batch_num = 0

        os.makedirs("plots/embedding-contrastive", exist_ok=True)
        threaded_loader = None
        timed_out = False
        metadata = {
            "base_model_name": "opcode-tokens-256",
            "margins": self.margin,
            "objective": objective.__class__.__name__,
            "plots": self.plot,
        }
        threaded_loader = objective.dataloader.iter(batch_size=32)

        try:
            for epoch in range(epochs):
                total_loss = 0
                total_correct = 0
                total_samples = 0
                steps = 0
                avg_loss = 0
                acc = 0

                pbar = tqdm(threaded_loader, desc=f"Epoch {epoch + 1}/{epochs}")
                timing_samples = 0
                time_forward = 0
                time_backward = 0
                time_step = 0

                try:
                    for index, batch in enumerate(pbar):
                        if time.time() - start_time > timeout_seconds:
                            pbar.close()
                            print(
                                f"Timeout reached ({timeout_minutes} min), stopping early..."
                            )
                            timed_out = True
                            break

                        t0 = time.perf_counter()
                        loss, correct, samples = objective.forward(batch)
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()

                        loss.backward()
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()

                        if index % 4 == 0 and index != 0:
                            objective.optimizer.step()
                            objective.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()

                        time_forward += t1 - t0
                        time_backward += t2 - t1
                        time_step += t3 - t2
                        timing_samples += 1

                        total_correct += correct
                        total_samples += samples
                        total_loss += loss.item()
                        steps += 1

                        avg_loss = total_loss / steps
                        acc = (
                            total_correct / total_samples * 100
                            if total_samples > 0
                            else 0
                        )
                        elapsed = (time.time() - start_time) / 60

                        pbar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            acc=f"{acc:.2f}%",
                            fwd=f"{1000 * time_forward / timing_samples:.0f}ms",
                            bwd=f"{1000 * time_backward / timing_samples:.0f}ms",
                            stp=f"{1000 * time_step / timing_samples:.0f}ms",
                            time=f"{elapsed:.1f}m",
                        )
                        batch_num += 1

                        if time.time() - self.last_checkpoint > 60 * 60:
                            self.checkpoints_tracker.checkpoint_files(
                                objective.checkpoint_files(),
                                Stats(
                                    loss_average=avg_loss,
                                    accuracy_pct=acc,
                                    runtime_seconds=time.time() - start_time,
                                    steps=batch_num,
                                    dataset=objective.dataloader.dataset_name,
                                    metadata=metadata,
                                ),
                            )
                            self.last_checkpoint = time.time()

                finally:
                    # Always cleanup the loader after each epoch
                    pass

                if steps > 0:
                    self.plot.record(
                        loss=(total_loss / steps),
                        accuracy=(total_correct / total_samples * 100),
                    )

                    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                    ax1.plot(
                        range(1, len(self.plot.losses) + 1), self.plot.losses, "b-o"
                    )
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.set_title("Training Loss")
                    ax1.grid(True)

                    ax2.plot(
                        range(1, len(self.plot.accuracies) + 1),
                        self.plot.accuracies,
                        "g-o",
                    )
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy (%)")
                    ax2.set_title("Training Accuracy")
                    ax2.grid(True)

                    plt.tight_layout()
                    plt.savefig("plots/embedding-contrastive/training.png", dpi=150)
                    plt.close()

                if timed_out:
                    print("Doing a checkpoint")
                    self.checkpoints_tracker.checkpoint_files(
                        objective.checkpoint_files(),
                        Stats(
                            loss_average=avg_loss,
                            accuracy_pct=acc,
                            runtime_seconds=time.time() - start_time,
                            steps=batch_num,
                            dataset=objective.dataloader.dataset_name,
                            metadata=metadata,
                        ),
                    )
                    print("Breaking out")
                    break

        finally:
            # Final cleanup in case of any exception
            if threaded_loader is not None:
                threaded_loader.cleanup()
            print("Shutting down .... ")
        print("Waiting for full shutdown")
        self.checkpoints_tracker._shutdown()
        print("DOne")


def train_model(epochs):
    # (model, _) = load_model_from_path(
    #    #"checkpoints/2026-01-11/20260111_115502/step_180841"
    # )
    # _, optimizer_state, stats = load_raw_from_path(
    #    #"checkpoints/2026-01-11/20260111_115502/step_180841"
    # )

    # dataloader = dataloader.iter(workers=2, batch_size=2)
    # print(next(iter(dataloader)))
    # dataloader.iter._start_epoch()
    # print(dataloader.iter._fetch_batch(0))
    # exit(0)
    #    for x in iter(dataloader):
    #        print(x)

    model, _ = load_best_model_from_checkpoint(
        target_dataset="opcode-tokens-256", max_age_days=32
    )
    plot = TrainingHistory()  # **stats["metadata"]["plots"])
    #   print(plot.accuracies)
    #   print(plot.losses)
    #    exit(0)
    # trainer.optimizer.load_state_dict(optimizer_state)
    objective = InfoObjectiveDataset(model, "cuda")
    trainer = EmbeddingTrainer(device="cuda", margin=0.6, plot=plot)
    trainer.train(objective, epochs=epochs, timeout_minutes=(48 * 60 * 2))


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
    project_head_state = torch.load(
        io.BytesIO(
            storage.load_bytes(
                "checkpoints/2026-01-18/20260118_195410/step_467974/proj_head.pt"
            )
        ),
        map_location=torch.device("cpu"),
    )
    model = InfoObjectiveDataset.load_from_storage(base_model, project_head_state)
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
            embeddings = torch.concat([model.embed(v) for v in doc_tensors], dim=0)

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

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve the model")
    serve_parser.add_argument("--port", type=int, default=8020, help="Port to serve on")

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
