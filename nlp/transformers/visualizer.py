import matplotlib.pyplot as plt


def plot_tensor(tensor, name):
    plt.imshow(tensor.numpy())
    plt.savefig(name)
    plt.clf()
