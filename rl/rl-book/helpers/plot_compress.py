from PIL import Image
import matplotlib.pyplot as plt

def savefig(filename, ):
    plt.savefig(filename)

    image = Image.open(filename)
    image.save(filename, quality=75, optimize=True)
