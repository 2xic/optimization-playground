from dataclasses import dataclass
import torch

@dataclass
class Config:
    z_size: int
    image_size: int    
    action_size: int
    device: torch.device
