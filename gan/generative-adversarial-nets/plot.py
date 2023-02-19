import matplotlib.pyplot as plt
import glob
from PIL import Image
files = glob.glob("imgs/*.png")

rows = 2
cols = 5
fig, axarr = plt.subplots(rows, cols)

step_size = len(files) // (rows * cols) + 1
fig.suptitle("Output over time")

for index, i in enumerate(range(0, len(files) - 1, step_size)):
    row = index // cols
    col = index % cols

    axarr[row][col].set_title(f"Epoch {i}")
    axarr[row][col].imshow(Image.open(f"imgs/img_{i}.png"), cmap='gray')
    axarr[row][col].axis('off')

plt.tight_layout()
plt.axis('off')
plt.savefig('results.png')
