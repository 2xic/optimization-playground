import os
from get_tokenized_shellcode import get_dataloader
from models.TransformerEncoder import TransformerEncoderModel, ModelWrapper
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from get_shellcode import get_shellcode
from flask import Flask, render_template
from flask import Flask
import json

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
    code = list(get_shellcode())
    names = list(map(lambda x: x[1], code))
    assembly_code = list(map(lambda x: x[0], code))
    print(names)
    output = model._forward(batch, model.embedding_a)

    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    embs = tsne.fit_transform(output.detach().numpy())

    ids = list(range(len(names)))
    X = embs[:, 0]
    Y = embs[:, 1]
    labels = names

    for (id, x, y, label, assembly) in zip(
        ids,
        X,
        Y,
        labels,
        assembly_code
    ):
        print((x, y, label, id))
        rows.append(
            {
                "id": id,
                "label": label,
                "x": float(x) * 100,
                "y": float(y) * 100,
                "fixed": True,
                "code": str(assembly)
            },
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
