import torch
import os
from torchvision.utils import save_image as torch_save_image
from tqdm import tqdm
from optimization_playground_shared.plot.Plot import Plot, Figure
import json

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAINING_STEPS = 5_000 #_000
VAE_TRAINING_STEPS = 3 * TRAINING_STEPS #TRAINING_STEPS
PREDICTOR_TRAINING_STEPS = TRAINING_STEPS #1_000 #TRAINING_STEPS # 10 # TRAINING_STEPS
CONTROLLER_TRAINING_STEPS =  TRAINING_STEPS

def save_model(model_class, optimizer_class, name):
    #print(f"Saving model {name}")
    torch.save({
        'model_state_dict': model_class.state_dict(),
        'optimizer_state_dict': optimizer_class.state_dict(),
    }, name)


def load_model(model_class, optimizer_class, name):
    if os.path.isfile(name):
        print(f"Loading model {name}")
        checkpoint = torch.load(name, map_location=DEVICE)
        model_class.load_state_dict(checkpoint['model_state_dict'])
        optimizer_class.load_state_dict(checkpoint['optimizer_state_dict'])


def save_image(tensor, file):
    dir = os.path.dirname(os.path.abspath(file))
    os.makedirs(dir, exist_ok=True)
    torch_save_image(tensor, file)


class LossTracker:
    def __init__(self, name):
        self.loss = []
        self.name = name

    def add_loss(self, item):
        self.loss.append(item)

    def save(self, x_scale=None, y_scale=None):
        file = self.name + '.png'
        file_json = self.name + '.json'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Loss": self.loss,
                    },
                    title="Loss",
                    x_axes_text="Epochs",
                    y_axes_text="Loss",
                    y_scale=y_scale,
                    x_scale=x_scale,
                    # x_scale='symlog'
                )
            ],
            name=file
        )
        with open(file_json, 'w') as file:
            json.dump({
                "Loss": self.loss,
            }, file)


class DebugInfo:
    def __init__(self, name):
        self.debug = []
        self.name = name

    def add_debug_info(self, **kwargs):
        self.debug.append(kwargs)

    def save(self):
        file = self.name + '.json'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

        with open(file, "w") as file:
            file.write(json.dumps(self.debug, indent=4))


class RunningAverage:
    def __init__(self) -> None:
        self.Q = 0
        self.N = 0

    def update(self, value):
        self.Q = self.Q + (value - self.Q) / (self.N + 1)
        self.N += 1
        return self.Q


class RewardTracker:
    def __init__(self, name):
        self.reward = []
        self.reference_reward = []
        self.name = name
        self.reward_avg = RunningAverage()
        self.reference_reward_avg = RunningAverage()

    def add_reward(self, item):
        self.reward.append(self.reward_avg.update(item))

    def add_reference_reward(self, item):
        self.reference_reward.append(self.reference_reward_avg.update(item))

    def save(self):
        file = self.name + '.png'
        file_json = self.name + '.json'
        dir = os.path.dirname(os.path.abspath(file))
        os.makedirs(dir, exist_ok=True)

        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Reward": (self.reward),
                        "Reference reward (untrained)": (self.reference_reward),
                    },
                    title="Reward",
                    x_axes_text="Epochs",
                    y_axes_text="Reward",
                )
            ],
            name=file
        )
        with open(file_json, 'w') as file:
            json.dump({
                "Reward": (self.reward),
                "Reference reward (untrained)": (self.reference_reward),
            }, file)
