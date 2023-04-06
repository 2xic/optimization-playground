from utils import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot():
    x, y, _, _ = load_dataset()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)
    
    df = pd.DataFrame()
    df['y'] = y
    df['pca-2d-one'] = pca_result[:,0]
    df['pca-2d-two'] = pca_result[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-2d-one", y="pca-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig('pca.png')

if __name__ == "__main__":
    plot()
