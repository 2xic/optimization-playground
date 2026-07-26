import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfidenceInterval:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @classmethod
    def create(cls, value):
        return cls(count=1, mean=value, m2=0.0)

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        return self

    @property
    def variance(self):
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self):
        return self.variance**0.5

    @property
    def lower_bound(self):
        return self.mean - self.std

    @property
    def upper_bound(self):
        return self.mean + self.std


class MinMaxAvgArray:
    def __init__(self):
        self.min_max_avg: list[ConfidenceInterval] = []

    def add(self, entries):
        is_new = len(self.min_max_avg) == 0
        # assert is_new or len(self.min_max_avg) == len(entries)
        for index, i in enumerate(entries):
            if is_new:
                self.min_max_avg.append(ConfidenceInterval.create(i))
            else:
                self.min_max_avg[index].update(i)

    def get_arrays(self):
        min = list(map(lambda x: x.lower_bound, self.min_max_avg))
        max = list(map(lambda x: x.upper_bound, self.min_max_avg))
        avg = list(map(lambda x: x.mean, self.min_max_avg))
        return min, max, avg

    def __len__(self):
        return len(self.min_max_avg)


@dataclass
class Results:
    accuracy: MinMaxAvgArray
    loss: MinMaxAvgArray
    step_accuracy: MinMaxAvgArray = field(default_factory=MinMaxAvgArray)
    step_loss: MinMaxAvgArray = field(default_factory=MinMaxAvgArray)
    epoch_at_step: list[int] = field(default_factory=list)
    step_val_accuracy: MinMaxAvgArray = field(default_factory=MinMaxAvgArray)
    step_val_loss: MinMaxAvgArray = field(default_factory=MinMaxAvgArray)
    val_at_step: list[int] = field(default_factory=list)

    @property
    def has_step_data(self):
        return len(self.step_accuracy) > 0 or len(self.step_loss) > 0


def running_average(data):
    running_sum = 0
    for i, value in enumerate(data, 1):
        running_sum += value
        if i > 0:
            running_avg = running_sum / i
        yield running_avg


def _plot_series(ax, key, color, primary: MinMaxAvgArray, epoch_markers: Optional[MinMaxAvgArray] = None, epoch_positions: Optional[list[int]] = None, smooth=False):
    (lo, hi, avg) = primary.get_arrays()
    if smooth:
        lo = list(running_average(lo))
        hi = list(running_average(hi))
        avg = list(running_average(avg))
    x = np.arange(len(avg))
    ax.plot(x, avg, color=color, label=key, alpha=0.6)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)
    if epoch_markers is not None and len(epoch_markers) > 0 and epoch_positions:
        (_, _, epoch_avg) = epoch_markers.get_arrays()
        if smooth:
            epoch_avg = list(running_average(epoch_avg))
        epoch_x = np.array(epoch_positions[:len(epoch_avg)])
        epoch_x = np.minimum(epoch_x, len(avg) - 1)
        ax.scatter(epoch_x, epoch_avg, color=color, marker="o", s=40, zorder=5)


def plot_accuracy_loss(results: dict[str, Results], file_path: str):
    items = list(results.values())
    if len(items[0].accuracy) == 1 and not items[0].has_step_data:
        plot_single_result_bar_chart(results, file_path)
        return

    has_steps = any(v.has_step_data for v in results.values())
    x_label = "Recorded step (sampled)" if has_steps else "Epoch"
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    colors = ["blue", "green", "red", "yellow", "orange", "purple"]

    for index, (key, value) in enumerate(results.items()):
        color = colors[index % len(colors)]
        if value.has_step_data:
            _plot_series(ax1, key, color, value.step_accuracy, epoch_markers=value.accuracy, epoch_positions=value.epoch_at_step)
            _plot_series(ax2, key, color, value.step_loss, epoch_markers=value.loss, epoch_positions=value.epoch_at_step, smooth=True)
        else:
            _plot_series(ax1, key, color, value.accuracy)
            _plot_series(ax2, key, color, value.loss, smooth=True)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.legend(loc="lower right")

    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.legend(loc="upper right")

    file_path = file_path.split(".")[0]
    print(f"Output: {file_path}.png")
    plt.savefig(f"{file_path}.png")
    plt.close("all")


def plot_scaling_laws(points: list[dict], file_path: str):
    import json
    pts = [p for p in points if p.get("val_loss") and p.get("flops")]
    pts.sort(key=lambda p: p["flops"])
    file_path = file_path.split(".")[0]
    with open(f"{file_path}.json", "w") as f:
        json.dump(pts, f, indent=2)
    if not pts:
        print(f"No scaling points for {file_path}")
        return

    flops = np.array([p["flops"] for p in pts])
    losses = np.array([p["val_loss"] for p in pts])

    frontier_x, frontier_y = [], []
    best = float("inf")
    for x, y in zip(flops, losses):
        if y < best:
            best = y
            frontier_x.append(x)
            frontier_y.append(y)

    _, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(flops, losses, s=60, color="blue", zorder=5)
    ax.plot(frontier_x, frontier_y, color="red", linestyle="--",
            label="compute-optimal frontier")
    for p in pts:
        params_m = p["params"] / 1e6
        ax.annotate(f"{p['label']} ({params_m:.1f}M)",
                    (p["flops"], p["val_loss"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    lx = np.log10(flops)
    if len(pts) >= 2 and np.ptp(lx) > 1e-6:
        ly = np.log10(losses)
        a, b = np.polyfit(lx, ly, 1)
        fit_x = np.array([flops.min(), flops.max()])
        fit_y = 10 ** (a * np.log10(fit_x) + b)
        ax.plot(fit_x, fit_y, color="green", alpha=0.5,
                label=f"fit: loss ~ C^{a:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs = 6 * N * D)")
    ax.set_ylabel("Val loss")
    ax.set_title("Scaling laws: val loss vs compute")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    print(f"Output: {file_path}.png")
    plt.savefig(f"{file_path}.png")
    plt.close("all")


def plot_single_result_bar_chart(results: dict[str, Results], file_path: str):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for key, value in results.items():
        acc_min, acc_max, acc_avg = value.accuracy.get_arrays()
        loss_min, loss_max, loss_avg = value.loss.get_arrays()

        # Use final epoch values
        final_acc = acc_avg[-1] if len(acc_avg) > 0 else 0
        final_loss = loss_avg[-1] if len(loss_avg) > 0 else 0

        # Bar chart with error bars
        ax1.bar(
            str(key),
            final_acc,
            yerr=[[final_acc - acc_min[-1]], [acc_max[-1] - final_acc]],
            capsize=5,
            alpha=0.7,
        )
        ax2.bar(
            str(key),
            final_loss,
            yerr=[[final_loss - loss_min[-1]], [loss_max[-1] - final_loss]],
            capsize=5,
            alpha=0.7,
        )

    ax1.set_title("Final Accuracy")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2.set_title("Final Loss")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

    file_path = file_path.split(".")[0]
    print(f"Output: {file_path}.png")
    plt.tight_layout()  # Prevent label cutoff
    plt.savefig(f"{file_path}.png")
    plt.close("all")
