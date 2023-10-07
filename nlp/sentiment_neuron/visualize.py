"""
We send in text from a simple web interface

We return images of the activation weights
"""
from flask import Flask, render_template
from model import Model
from vocab import Vocab
from optimization_playground_shared.plot.Plot import Plot, Image
import torch
import base64

app = Flask(__name__)


class SaveOutput:
    def __init__(self):
        self.outputs = []
        self.firing_scores = {}

    def add_firing_value(self, index, value):
        if not index in self.firing_scores:
            self.firing_scores[index] = [value]
        else:
            self.firing_scores[index].append(value)

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


def single_forward(text):
    model = Model.load()
    vocab = Vocab().load()
    saved_output_1 = SaveOutput()
    saved_output_2 = SaveOutput()
    saved_output_3 = SaveOutput()
    saved_output_4 = SaveOutput()
    for i, j in zip([
        model.activation_1,
        model.activation_2,
        model.activation_3,
        model.activation_4,
    ], [
        saved_output_1,
        saved_output_2,
        saved_output_3,
        saved_output_4,
    ]):
        i.register_forward_hook(j)

    _ = model.forward_feed(vocab.get_encoded(text))
    outputs = [

    ]
    # create one neuron layer map
    predictions = [
        saved_output_1.get_and_clear(),  # pred1, pred2, pred3
        saved_output_2.get_and_clear(),
        saved_output_3.get_and_clear(),
        saved_output_4.get_and_clear(),
    ]
    prediction_index = 0
    done = False
    outputs = []
    while not done:
        current_stream = []
        for i in predictions:
            if prediction_index < len(i):
                current_stream.append(i[prediction_index])
            else:
                done = True
                break
        if not done:
            outputs.append(current_stream)
        prediction_index += 1
   # print(outputs)
    plot_outputs = []
    input_text = ""
    for prediction_stream, char in zip(outputs, text):
        layers = []
        input_text += char
        for index, i in enumerate(prediction_stream):
            tensor = i.detach().to(torch.device('cpu')).numpy()
            layers.append(
                Image(
                    image=tensor,
                    title=f'layer_{index}'
                )
            )
        plot = Plot().plot_image(layers, f'inference.png')
        with open(plot, "rb") as file:
            data = base64.b64encode(
                # slice the 0x
                file.read()
            ).decode("ascii")
            plot_outputs.append(f"<h1>{input_text}</h1>")
            plot_outputs.append(f"<img src='data:image/png;base64,{data}'/>")
    return "<br>".join(plot_outputs)

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/forward/<text>')
def forward(text):
    return single_forward(text)


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=8085)
