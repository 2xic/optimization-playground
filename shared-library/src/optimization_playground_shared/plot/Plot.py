"""
Basic plotting util to simplify life
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict
import torch

@dataclass
class Figure:
    plots: Dict[str, Any]    
    title: str
    x_axes_text: str=None
    y_axes_text: str=None
    y_min: int=None
    y_max: int=None

class Plot:
    def __init__(self):
        pass
    
    def plot(self, figures, name):
        n_cols = len(figures)
        n_rows = 1
        
        _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5))
        for index, i in enumerate(figures):
            for legend, value in i.plots.items():
                # make sure the tensor is on the cpu
                value = list(map(self._convert_list, value))
                axes[index].plot(value, label=legend)
            axes[index].set_ylim(i.y_min, i.y_max)
            axes[index].set_title(i.title)
            axes[index].set_xlabel(i.x_axes_text)
            axes[index].set_ylabel(i.y_axes_text)
            axes[index].legend(loc="upper left")
        plt.savefig(name)

    def _convert_list(self, entry):
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu()
        return entry
