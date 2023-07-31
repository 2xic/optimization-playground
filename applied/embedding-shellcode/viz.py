import os
from get_tokenized_shellcode import get_dataloader
from models.TransformerEncoder import TransformerEncoderModel
import torch
from sklearn.manifold import TSNE
from flask import Flask, render_template
from flask import Flask
import json
from eval import get_2d_embeddings

app = Flask(__name__)

rows = []


@app.route('/data')
def data():
    return json.dumps(rows)

@app.route('/')
def index():
    return render_template('index.html')


def test():
    _, dataset = get_dataloader()

    model = TransformerEncoderModel(
        n_token=dataset.n_tokens,
        device=torch.device('cpu'),
        n_layers=1,
    )
    data = torch.load('model.pkt')['model']
    model.load_state_dict(data)

    batch = dataset.program_tensor
    output = model._forward(batch, model.embedding_a)

    rows = get_2d_embeddings(
        output.detach().numpy()
    )

    with open("dump.json", "w") as file:
        json.dump({
            "rows": rows,
        }, file)

if __name__ == "__main__":
    if os.path.isfile("dump.json"):
        with open("dump.json", "r") as file:
            rows = json.load(file)
    else:
        test()

    app.run(
        host='0.0.0.0'
    )
