import torch
from torchvision.models import vgg19
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torchvision
import os


def generate_image(content_image_path, style_image_path, output_path):
    if os.path.isfile(output_path):
        return None
    input_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                             std=[1, 1, 1]),
    ])
    output_transforms = transforms.Compose([
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                             std=[1, 1, 1])
    ])

    content_image = input_transforms(
        Image.open(content_image_path)).unsqueeze(0)
    style_image = input_transforms(Image.open(style_image_path)).unsqueeze(0)
    assert content_image.shape[1] == style_image.shape[1], "Must have same height"
    original_content_image_shape = content_image.shape
    # Do the resize of width dynamically
    transform = transforms.Pad(
        (0, 0, abs(content_image.shape[3] - style_image.shape[3]), 0))

    def has_smaller_width(x, y): return x.shape[3] < y.shape[3]
    content_image = transform(content_image) if has_smaller_width(
        content_image, style_image) else content_image
    style_image = transform(style_image) if has_smaller_width(
        style_image, content_image) else style_image
    assert content_image.shape[3] == style_image.shape[3], "Must have same width"

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

    content_output_features = None
    with torch.no_grad():
        _ = vgg_model(content_image)
        content_output_features = output.get_and_clear()

    style_output_features = None
    with torch.no_grad():
        _ = vgg_model(style_image)
        style_output_features = output.get_and_clear()

    def get_style_loss(model_output_features):
        loss = 0
        for index, (x, y) in enumerate(zip(model_output_features, style_output_features)):
            _, c, h, w = x.shape

            generated = x.reshape((x.shape[-1], -1))
            style = y.reshape((y.shape[-1], -1))
            # print(x)
            # print(y)
            # print("====")
            # gram matrix
            style_gg = torch.matmul(generated, generated.T)
            style_sg = torch.matmul(style, style.T)

            scale = 1e3/(index + 1)**2
            layer_loss = scale * (1 / (4 * (c) ** 2 * (h * w) ** 2)) * \
                ((style_sg - style_gg) ** 2).sum()
            loss += layer_loss
        return loss

    def get_content_loss(model_output_features):
        loss = 0
        # usually -1
        for i in [0]:
            last_content_feature = content_output_features[i]
            last_generated_feature = model_output_features[i]
            _, c, h, w = last_content_feature.shape
            content_l = (1/(4 * h * w * c)) * \
                ((last_content_feature - last_generated_feature) ** 2).sum()
            loss += content_l
        return content_l

    # optimization steps
    generated_image = torch.rand(content_image.shape)
    generated_image.requires_grad_(True)
    optimizer = optim.LBFGS([generated_image])

    for epoch in range(100):
        # make sure we stay within the correct range
        with torch.no_grad():
            generated_image.clamp_(0, 1)

        def closure():
            # need to do one forward
            _ = vgg_model(generated_image)
            generated_features_features = output.get_and_clear()
            optimizer.zero_grad()
            loss = 0.3 * get_style_loss(generated_features_features) + \
                10 * get_content_loss(generated_features_features)

            if loss == torch.inf:
                return 0
            loss.backward()
            # I don't want the style to go out of bounds
            torch.nn.utils.clip_grad_norm_([generated_image], max_norm=1_00)
           # torch.nn.utils.clip_grad_norm_(optimizer.param_groups, 1.0)
            print(f"Epoch: {epoch}, loss: {loss.item()}")
            return loss
        optimizer.step(closure)
    #    print(f"Epoch: {epoch}, loss: {loss.item()}")

        if epoch % 10 == 0:
            torchvision.utils.save_image(
                # reshape the width to the original width to not have artifacts
                output_transforms(generated_image)[
                    :, :, :, :original_content_image_shape[3]],
                output_path
            )


if __name__ == "__main__":
    generate_image(
        "images/mona_lisa.jpg",
        "images/the_scream.jpg",
        "images/mona_lisa_style_of_scream.png",
    )
    generate_image(
        "images/mona_lisa.jpg",
        "images/portrait_of_picasso.jpg",
        "images/mona_lisa_style_of_picasso_portrait.png",
    )
