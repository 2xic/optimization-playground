import matplotlib.pyplot as plt
import glob
import os
import json
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

for output_name, field, title in [
    ['train_plot.png', 'train_acc', 'Training accuracy'],
    ['test_plot.png', "test_acc", 'Testing accuracy']
]:
    line_plot = {}
    size = {

    }
    min_size = float('inf')
    max_size = float(0)
    max_width = 0
    for i in glob.glob("./data/*.json"):
        name = int(
            os.path.basename(i).replace(".json", "")
        )
        # did only train up to 90 in width sequence
        if name in [128]:
            continue
        with open(i, "r") as file:
            content = json.loads(file.read())
            size[name] = content[field]
            line_plot[name] = content[field]
            min_size = min(min_size, len(size[name]))
            max_size = max(max_size, len(size[name]) + 1)
        max_width = max(name + 1, max_width)

    for i in size:
        size[i] = size[i][min_size - 1]
    points = size.items()
    fig, ax = plt.subplots()

    ax.scatter(
        list(map(lambda x: x[0], points)),
        list(map(lambda x: x[1], points))
    )
    ax.set_xlabel("Resnet width")
    ax.set_ylabel("Accuracy")
    plt.title(title)
    plt.savefig("./plots/" + output_name)
    plt.clf()

    fig, ax = plt.subplots()
    for key, value in sorted(line_plot.items(), key=lambda x: (x[1][-1]), reverse=True)[:10]:
        plt.plot(value, label=key)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.title(f"{title} (top 10)")
    plt.savefig("./plots/line_" + output_name)
    plt.clf()

    ax = plt.axes()
    step_size = 100
    img = np.zeros((max_width, (max_size // step_size) + 1))
    for i in line_plot:
        y = line_plot[i][::step_size]
        x = int(i)
        img[x, :len(y)] = y
    im = plt.imshow(img.T, cmap='jet')
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: int(
            x * step_size
        )))
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Resnet width")
    plt.title("Epoch wise double decent")

    # Thank you https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax)
    plt.savefig('./plots/epoch_wise_' + output_name)
    plt.clf()
