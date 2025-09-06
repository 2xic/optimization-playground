import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from experiments import (
    create_default_config,
    TransformerLayerType,
    Datasets,
)
from training.model import Model
import random


def create_triplets_with_hard_negatives(bytecode_sequences):
    triplets = []

    for i, seq in enumerate(bytecode_sequences):
        for j, anchor in enumerate(seq):
            # Positive: nearby window from same sequence
            positives = [
                seq[k] for k in range(max(0, j - 3), min(len(seq), j + 4)) if k != j
            ]

            # Negative: random window from different sequence
            other_sequences = [
                s for idx, s in enumerate(bytecode_sequences) if idx != i
            ]
            negatives = [window for seq in other_sequences for window in seq]

            if positives and negatives:
                triplets.append(
                    {
                        "anchor": anchor,
                        "positive": random.choice(positives),
                        "negative": random.choice(negatives),
                    }
                )

    return triplets


class ContrastiveTransformer(nn.Module):
    def __init__(self, transformer_model, embedding_dim):
        super().__init__()
        self.transformer = transformer_model
        self.pooler = nn.Linear(transformer_model.config.vocab_size, embedding_dim)

    def forward(self, input_ids):
        outputs = self.transformer(input_ids)
        cls_embedding = outputs[:, 0, :]
        sentence_embedding = self.pooler(cls_embedding)
        return F.normalize(sentence_embedding, p=2, dim=1)


def contrastive_loss(anchor_emb, pos_emb, neg_emb, temperature=0.07):
    """Calculate contrastive loss"""
    pos_sim = torch.sum(anchor_emb * pos_emb, dim=1) / temperature
    neg_sim = torch.sum(anchor_emb * neg_emb, dim=1) / temperature

    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def prepare_batch(batch_triplets):
    anchor_ids = []
    pos_ids = []
    neg_ids = []

    for triplet in batch_triplets:
        # Tokenize anchor, positive, negative
        anchor_tokens = triplet["anchor"]
        pos_tokens = triplet["positive"]
        neg_tokens = triplet["negative"]

        anchor_ids.append(anchor_tokens)
        pos_ids.append(pos_tokens)
        neg_ids.append(neg_tokens)

    return {
        "anchor_ids": torch.stack(anchor_ids),
        "pos_ids": torch.stack(pos_ids),
        "neg_ids": torch.stack(neg_ids),
    }


def train_loop_test():
    device = torch.device("cuda:1")
    dataset = Datasets.tiny_evm_bytecode()
    model_config = create_default_config(
        dataset,
    ).with_transformer_layer(TransformerLayerType.BERT)

    base_model = Model(model_config)
    contrastive_model = ContrastiveTransformer(base_model, 512).to(device)
    optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-4)

    for index, i in enumerate(
        dataset.dataset.get_file_content_tokenized(sequence_size=512)
    ):
        chunks = list(
            torch.chunk(i, chunks=int(i.shape[-1] / model_config.sequence_length))
        )
        triplets = create_triplets_with_hard_negatives(chunks)
        batch_size = 32

        anchor = torch.zeros(
            (len(triplets) // batch_size, batch_size), device=device
        ).long()
        pos = torch.zeros(
            (len(triplets) // batch_size, batch_size), device=device
        ).long()
        neg = torch.zeros(
            (len(triplets) // batch_size, batch_size), device=device
        ).long()

        for batch_index, i in enumerate(range(0, len(triplets), batch_size)):
            batch_triplets = triplets[i : i + batch_size]
            batch = prepare_batch(batch_triplets)

            anchor_ids = batch["anchor_ids"]
            pos_ids = batch["pos_ids"]
            neg_ids = batch["neg_ids"]

            anchor[batch_index] = anchor_ids
            pos[batch_index] = pos_ids
            neg[batch_index] = neg_ids

        anchor_emb = contrastive_model(anchor)
        pos_emb = contrastive_model(pos)
        neg_emb = contrastive_model(neg)

        # Calculate loss
        loss = contrastive_loss(anchor_emb, pos_emb, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print("Index: {index}, loss: {loss}".format(index=index, loss=loss.item()))
        if index % 1000 == 0:
            torch.save(contrastive_model.state_dict(), "evm_contrastive_model_debug")


if __name__ == "__main__":
    train_loop_test()
