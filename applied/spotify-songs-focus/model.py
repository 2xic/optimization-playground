import glob 
from music_features import process_audio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
np.random.seed(42)

feature_vectors = []
for i in tqdm(glob.glob("audio_features_dataset/*.mp3")):
    feature_vectors.append(process_audio(i, None))
feature_vectors = np.asarray(feature_vectors)
print(feature_vectors.shape)

tsne = TSNE(n_components=2, random_state=42)
embedded_vectors = tsne.fit_transform(feature_vectors)
plt.figure(figsize=(10, 8))
plt.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1], cmap='tab10', s=50, alpha=0.8)
plt.grid(True)
plt.savefig('features.png')
