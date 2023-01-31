import torch
from helpers.basic_model import Net as LinearModel
from helpers.basic_conv_model import Net as ConvModel
from helpers.timer import TimeIt

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


for (create_model, shape_creator) in models_input_shape:
    non_jit = TimeIt('non_jit')
    jit = TimeIt('jit')

    model = create_model()

    jit_model = create_model()
    jit_model.load_state_dict(model.state_dict())
    jit_model = torch.jit.optimize_for_inference(torch.jit.script(
        jit_model
    ))

#    jit_model = torch.compile(model, mode="reduce-overhead")
    batch_size = 1024
    X = torch.rand(shape_creator(batch_size,))

    for i in range(100):
        with non_jit() as x:
            model(X)

        with jit() as x:
            jit_model(X)

    print(sum(non_jit.times))
    print(sum(jit.times))
    print("")
