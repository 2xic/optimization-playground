import torch.nn.functional as F
import torch
from training.model import Model
import torch
import matplotlib.pyplot as plt
import time
import torch
from utils.web_dataloader import WebDataloader
import os
from dotenv import load_dotenv
from utils.checkpoints import StorageBoxCheckpoint, Stats
from datetime import datetime
from tqdm import tqdm
from utils.load_mode_from_checkpoint import load_best_model_from_checkpoint

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
        model: Model,
        dataloader: WebDataloader,
        lr=2e-5,
        margin=0.2,
        device="cuda",
    ):
        self.model: Model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.margin = margin
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoints_tracker = StorageBoxCheckpoint(
            run_id=run_id,
            host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
            username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
            password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
        )
        self.last_checkpoint = time.time()

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

    def train(self, epochs=3, timeout_minutes=60):
        epoch_losses = []
        epoch_accuracies = []
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        batch_num = 0

        os.makedirs("plots/embedding-contrastive", exist_ok=True)
        threaded_loader = None
        timed_out = False
        use_triplet_loss = False
        metadata = {
            "base_model_name": "opcode-tokens-256",
            "margins": self.margin,
            "use_triplet_loss": use_triplet_loss,
        }

        try:
            for epoch in range(epochs):
                total_loss = 0
                total_correct = 0
                total_samples = 0
                steps = 0
                avg_loss = 0
                acc = 0

                threaded_loader = self.dataloader.iter()
                pbar = tqdm(threaded_loader, desc=f"Epoch {epoch + 1}/{epochs}")

                try:
                    for batch in pbar:
                        if time.time() - start_time > timeout_seconds:
                            pbar.close()
                            print(
                                f"Timeout reached ({timeout_minutes} min), stopping early..."
                            )
                            timed_out = True
                            break

                        a = batch["ref_tokens"].to(self.device)
                        p = batch["pos_tokens"].to(self.device)
                        n = batch["neg_tokens"].to(self.device)

                        if a.numel() == 0:
                            continue

                        anchor_emb = self.embed(a)
                        pos_emb = self.embed(p)

                        if use_triplet_loss:
                            neg_emb = self.embed(n)
                            loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
                            pos_dist = 1 - F.cosine_similarity(anchor_emb, pos_emb)
                            neg_dist = 1 - F.cosine_similarity(anchor_emb, neg_emb)
                            correct = (pos_dist < neg_dist).sum().item()
                        else:
                            loss, correct = self.in_batch_contrastive_loss(
                                anchor_emb, pos_emb
                            )
                            total_correct += correct

                        total_samples += a.size(0)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

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
                            time=f"{elapsed:.1f}m",
                        )
                        batch_num += 1

                        if time.time() - self.last_checkpoint > 60 * 60:
                            self.checkpoints_tracker.checkpoint(
                                self.model,
                                self.optimizer,
                                self.model.config,
                                Stats(
                                    loss_average=avg_loss,
                                    accuracy_pct=acc,
                                    runtime_seconds=time.time() - start_time,
                                    steps=batch_num,
                                    dataset="etherscan-similarity-256-v2",
                                    metadata=metadata,
                                ),
                            )
                            self.last_checkpoint = time.time()

                finally:
                    # Always cleanup the loader after each epoch
                    if threaded_loader is not None:
                        threaded_loader.iter.cleanup()
                        threaded_loader = None

                if steps > 0:
                    epoch_losses.append(total_loss / steps)
                    epoch_accuracies.append(total_correct / total_samples * 100)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                    ax1.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-o")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.set_title("Training Loss")
                    ax1.grid(True)

                    ax2.plot(
                        range(1, len(epoch_accuracies) + 1), epoch_accuracies, "g-o"
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
                    self.checkpoints_tracker.checkpoint(
                        self.model,
                        self.optimizer,
                        self.model.config,
                        Stats(
                            loss_average=avg_loss,
                            accuracy_pct=acc,
                            runtime_seconds=time.time() - start_time,
                            steps=batch_num,
                            dataset="etherscan-similarity-256-v2",
                            metadata=metadata,
                        ),
                    )
                    print("Breaking out")
                    break

        finally:
            # Final cleanup in case of any exception
            if threaded_loader is not None:
                threaded_loader.iter.cleanup()
            print("Shutting down .... ")
        print("Waiting for full shutdown")
        self.checkpoints_tracker._shutdown()
        print("DOne")


def fine_tune_model():
    model, _ = load_best_model_from_checkpoint(
        target_dataset="opcode-tokens-256", max_age_days=14
    )
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataset_name="etherscan-similarity-256-v2",
        columns=["ref_tokens", "pos_tokens", "neg_tokens"],
    )
    trainer = EmbeddingTrainer(
        model=model, dataloader=dataloader, device="cuda", margin=0.6
    )
    trainer.train(epochs=1_000, timeout_minutes=360)


if __name__ == "__main__":
    fine_tune_model()
    print("The end")
