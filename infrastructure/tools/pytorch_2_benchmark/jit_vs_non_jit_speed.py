import torch
from helpers.basic_model import LinearModel
from helpers.basic_conv_model import ConvModel
from helpers.timer import TimeIt
import json

models_input_shape = (
    [
        lambda: LinearModel(256).eval(),
        lambda batch_size: (batch_size, 256),
    ],
    [
        lambda: ConvModel().eval(),
        lambda batch_size: (batch_size, 3, 32, 32),
    ]
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = {}

for (create_model, shape_creator) in models_input_shape:
    non_jit = TimeIt('non_jit')
    jit = TimeIt('jit')

    model = create_model().to(device)
    jit_model = create_model().to(device)

    jit_model.load_state_dict(model.state_dict())
    jit_model = torch.jit.optimize_for_inference(torch.jit.script(
        jit_model
    ))

    batch_size = 1024
    X = torch.rand(shape_creator(batch_size,)).to(device)

    for i in range(100):
        with non_jit() as x:
            model(X)

        with jit() as x:
            jit_model(X)

    data[model.__class__.__name__] = {
        "non_jit": sum(non_jit.times),
        "jit": sum(jit.times)
    }
    print(data)

with open(f"torch_{torch.__version__}.json", "w") as file:
    json.dump(
        data,
        file
    )
