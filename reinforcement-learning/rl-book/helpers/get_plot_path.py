import os

def get_plot_path(file):
    name = os.path.basename(os.path.abspath(file)).replace(".py", ".png")
    path = os.path.abspath(os.path.dirname(file))

    dirname = os.path.join(
        path,
        "plot",
    )
    os.makedirs(dirname, exist_ok=True)

    return os.path.join(
        dirname,
        name
    )
