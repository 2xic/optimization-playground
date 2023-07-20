"""
Basic plotting util to simplify life
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


@dataclass
class Figure:
    plots: Dict[str, Any]
    title: str
    x_axes_text: Optional[str] = None
    y_axes_text: Optional[str] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    x_scale: Optional[str] = None


@dataclass
class Image:
    image: Any
    title: str


class Plot:
    def __init__(self):
        pass

    def plot_image(self, images, name):
        n_cols = len(images)
        n_rows = 1

        _, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                               figsize=(5 * n_cols, 5))
        if type(axes) != list and not isinstance(axes, np.ndarray):
            axes = [axes]

        for index, i in enumerate(images):
            image = i.image

            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image.reshape(image.shape[1:])

            if len(image.shape) == 2:
                axes[index].imshow(image, cmap='gray')
            else:
                axes[index].imshow(image)
            axes[index].set_title(i.title)
            axes[index].axis('off')

        plt.savefig(name)

    def plot_figures(self, figures, name):
        n_cols = len(figures)
        n_rows = 1

        _, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                               figsize=(5 * n_cols, 5))
     #   print(type(axes))
        if n_cols == 1 and n_rows == 1:
            axes = [axes]

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
            axes[index].set_xscale('symlog')

        plt.savefig(name)
        plt.clf()
        plt.cla()
        plt.close('all')

    def _convert_list(self, entry):
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu()
        return entry
