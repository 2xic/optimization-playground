"""
Same setup as the style transfer code
"""
import torch
from torchvision.models import vgg19
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torchvision

def generate_image(output_path):
    output_transforms = transforms.Compose([
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                             std=[1, 1, 1])
    ])

    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, _, __, module_out):
            self.outputs.append(module_out)

        def clear(self):
            self.outputs = []

        def get_and_clear(self):
            output = [
                i.clone()
                for i in self.outputs
            ]
            self.outputs.clear()
            return output

    vgg_model = vgg19(pretrained=True)
    vgg_model.eval()
    vgg_model.requires_grad_(False)

    output = SaveOutput()
    for i in [
        vgg_model.features[0],
        vgg_model.features[2],
        vgg_model.features[5],
        vgg_model.features[7],
        vgg_model.features[10],
    ]:
        assert isinstance(i, torch.nn.Conv2d)
        i.register_forward_hook(output)


    def get_content_loss(model_output_features):
        loss = 0
        # usually -1
        for i in [0]:
            last_generated_feature = model_output_features[i]
            _, c, h, w = last_generated_feature.shape
            # I want the first 1 neurons !!! to fire
            content_l = (1/(4 * h * w * c)) * \
                (last_generated_feature[0, :] ** 2).sum()
            loss += content_l
        return content_l

    # optimization steps
    generated_image = torch.rand((1, 3, 256, 256))
    generated_image.requires_grad_(True)
    optimizer = optim.LBFGS([generated_image])

    for epoch in range(100):
        # make sure we stay within the correct range
        with torch.no_grad():
            generated_image.clamp_(0, 1)

        def closure():
            # need to do one forward
            class_output = vgg_model(generated_image)
            generated_features_features = output.get_and_clear()
            optimizer.zero_grad()
            # first image of imagenet is fish. I want to see fish like image.
            loss = 10 * get_content_loss(generated_features_features) + (
                class_output[1:].sum()
            )

            if loss == torch.inf:
                return 0
            loss.backward()
            # I don't want the style to go out of bounds
            torch.nn.utils.clip_grad_norm_([generated_image], max_norm=1_00)
            print(f"Epoch: {epoch}, loss: {loss.item()}")
            return loss
        optimizer.step(closure)

        if epoch % 10 == 0:
            torchvision.utils.save_image(
                # reshape the width to the original width to not have artifacts
                output_transforms(generated_image), # [:, :, :, :original_content_image_shape[3]]
                output_path
            )


if __name__ == "__main__":
   generate_image("viz.png")
