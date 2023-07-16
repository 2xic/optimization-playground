from optimization_playground_shared.dataloaders.Mnist import get_dataloader
import torch
from optimization_playground_shared.plot.Plot import Plot

(train, _) = get_dataloader()

class ImageOperations:
    def __init__(self):
        pass

    # equation 8 in the paper
    def D(self, image):
        # c_1 and c_2 should be fitted based on the dataset
        c_1 = 3
        c_2 = 2
        m = 2
        F = self.f(image)

        return (c_1) / (torch.abs(c_2 + F) ** m)

    # equation 9 in the paper
    def f(self, image):
        if image.shape == (1, 28, 28):
            image = image.reshape((28, 28))
        assert len(image.shape) == 2, "Should be 2d image"
        (rows, columns) = image.shape

        sobel_filtered_image = torch.zeros(image.shape)
        Gx = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = torch.sum(torch.multiply(Gx, image[i:i + 3, j:j + 3])) 
                gy = torch.sum(torch.multiply(Gy, image[i:i + 3, j:j + 3]))
                sobel_filtered_image[i + 1, j + 1] = torch.sqrt(gx ** 2 + gy ** 2)  
        return sobel_filtered_image

if __name__ == "__main__":
    (images, _) = next(iter(train))
    image = images[0]
    print(image.shape)

    F = ImageOperations().f(image)
    D = ImageOperations().f(image)

    Plot().plot_image([
        image,
        F,
        D,
    ], 'dft.png')
