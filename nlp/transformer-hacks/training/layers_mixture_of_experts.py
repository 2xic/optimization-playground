"""
https://medium.com/@baicenxiao/harnessing-the-power-of-mixture-of-experts-in-transformers-4140502e1c1e
https://dataturbo.medium.com/key-techniques-behind-deepseek-models-10x-efficiency-1-moe-9bd2534987c8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Config, Model
from .trainer import train
from utils.transformer_dataset import XorDataset


class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.layer(x.float()), dim=-1)


class MoE(nn.Module):
    def __init__(self, config: Config, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Model(config) for _ in range(num_experts)])
        self.config = config
        self.router = Router(config.sequence_length, num_experts)
        self.top_k = top_k

    def forward(self, x):
        batch_size = x.size(0)
        routing_weights = self.router(x)

        topk_vals, topk_indices = torch.topk(routing_weights, self.top_k, dim=1)
        topk_vals_normalized = topk_vals / topk_vals.sum(dim=1, keepdim=True)

        outputs = torch.zeros(
            batch_size,
            self.config.sequence_length,
            self.config.vocab_size,
            device=x.device,
        )

        for i, expert in enumerate(self.experts):
            expert_mask = (topk_indices == i).float()
            if expert_mask.any():
                expert_mask = expert_mask.unsqueeze(-1).expand(-1, -1, x.size(1))
                inputs_to_expert = x.unsqueeze(1).repeat(1, self.top_k, 1) * expert_mask
                inputs_to_expert = inputs_to_expert.view(-1, x.size(1))
                expert_outputs = expert(inputs_to_expert.long()).view(
                    batch_size, self.top_k, -1
                )

                # Weight outputs by normalized routing probability and sum across selected experts
                weighted_expert_outputs = (
                    expert_outputs * topk_vals_normalized.unsqueeze(-1)
                )
                weighted_expert_outputs = weighted_expert_outputs.sum(dim=1).reshape(
                    (batch_size, self.config.sequence_length, self.config.vocab_size)
                )
                outputs += weighted_expert_outputs
        return outputs


if __name__ == "__main__":
    (_, accuracy, loss) = train(XorDataset(), create_model=lambda x: MoE(x))
    print(loss)
