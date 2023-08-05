"""
Basic plotting util to simplify life
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
from typing import List
import os

@dataclass
class Figure:
    plots: Dict[str, Any]
    title: str
    x_axes_text: Optional[str] = None
    y_axes_text: Optional[str] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    x_scale: Optional[str] = None
    y_scale: Optional[str] = None

@dataclass
class Image:
    image: Any
    title: Optional[str] = None


@dataclass
class ScatterEntry:
    X: List[int]
    y: List[int]

@dataclass
class Scatter:
    plots: Dict[str, ScatterEntry]
    title: Optional[str] = None

class Plot:
    def __init__(self):
        pass

    def plot_image(self, images, name, row_direction=True):
        n_cols = len(images)
        n_rows = 1

        if not row_direction:
            n_cols, n_rows = n_rows, n_cols

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
    
            if i.title is not None:
                axes[index].set_title(i.title)
            axes[index].axis('off')

        self._create_output_folder(name)
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
            if i.x_scale is not None:
                axes[index].set_xscale(i.x_scale)
            if i.y_scale is not None:
                axes[index].set_yscale(i.y_scale)
            # axes[index].set_ylim(bottom=0)

        self._create_output_folder(name)
        plt.savefig(name)
        plt.clf()
        plt.cla()
        plt.close('all')

    def plot_scatter(self, figure: Scatter, name: str):
        for legend, value in figure.plots.items():
            plt.scatter(value.X, value.y, label=legend)

        plt.legend(loc="upper left")
        plt.title(figure.title)
        plt.savefig(name)
        plt.clf()
        plt.cla()
        plt.close('all')

    def _convert_list(self, entry):
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu()
        return entry

    def _create_output_folder(self, name):
        dirname = os.path.dirname(os.path.abspath(name))
        os.makedirs(dirname, exist_ok=True)

