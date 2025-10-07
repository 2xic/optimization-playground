import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def simple_temperature_sampling(values: torch.Tensor, temperature: float = 0.95):
    scaled_logits = values / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    dist = Categorical(probs=probs)
    sampled_values = dist.sample()
    return sampled_values


def simple_sampling(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    token_idx = torch.multinomial(probs, num_samples=1)
    return token_idx.squeeze(-1)


def temperature_sampling_old(logits, temperature=0.95, top_k=10, top_p=0.6):
    batch_size, vocab_size = logits.shape
    if temperature == 0:
        tokens = torch.argmax(logits, dim=-1)
        return tokens
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        top_values, _ = torch.topk(probs, top_k, dim=-1)
        threshold_values = top_values[:, -1].unsqueeze(-1)
        mask = probs >= threshold_values
        probs = probs * mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        for batch_idx in range(batch_size):
            indices_to_remove = sorted_indices_to_remove[batch_idx].scatter(
                0, sorted_indices[batch_idx], sorted_indices_to_remove[batch_idx]
            )
            probs[batch_idx][indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return tokens


def temperature_sampling(logits, temperature=0.95, top_k=10, top_p=0.6):
    _, vocab_size = logits.shape
    if temperature == 0:
        tokens = torch.argmax(logits, dim=-1)
        return tokens
    assert top_k > 0
    assert top_p > 0

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    top_k = min(top_k, vocab_size)
    top_values, top_indices = torch.topk(probs, top_k, dim=-1)

    cumulative_probs = torch.cumsum(top_values, dim=-1)
    cutoff_mask = cumulative_probs > top_p

    cutoff_mask = cumulative_probs > top_p

    filtered_probs = top_values.clone()
    cutoff_mask[..., 0] = False

    probs_sum = filtered_probs.sum(dim=-1, keepdim=True)
    probs_sum = torch.clamp(probs_sum, min=1e-9)
    filtered_probs = filtered_probs / probs_sum

    selected_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    tokens = top_indices.gather(-1, selected_idx.unsqueeze(-1)).squeeze(-1)

    return tokens


def argmax_sampling(values: torch.Tensor):
    return values.argmax(dim=-1)
