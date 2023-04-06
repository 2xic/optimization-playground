from sklearn.manifold import TSNE
from utils import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot():
    x, y, _, _ = load_dataset()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=3000)
    tsne_results = tsne.fit_transform(x)
    
    df = pd.DataFrame()
    df['y'] = y
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig('t_sne.png')


if __name__ == "__main__":
    plot()
