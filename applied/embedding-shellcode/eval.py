"""
Eval script to make sure the output is reasonable
"""

import os
import json
from optimization_playground_shared.clustering.elbow_method import ElbowMethod
import os
from sklearn.manifold import TSNE
import json
from get_tokenized_shellcode import get_current_dataset
from output_format import OutputFormat
from typing import List

def get_cluster_count(rows):
    vector = [
        (i["x"], i["y"])
        for i in rows["rows"]
    ]
    output = ElbowMethod().fit(
        vector
    )
    return output

def get_2d_embeddings(prediction):
    dataloader = get_current_dataset().dataloader.get_shellcode()
    rows = []
    code: List[OutputFormat] = list(dataloader)
    names = list(map(lambda x: x.name, code))
    assembly_code = list(map(lambda x: x.code, code))
    internal_ids = list(map(lambda x: x.id, code))

    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine", init='random', learning_rate='auto')
    embs = tsne.fit_transform(prediction)

    ids = list(range(len(names)))
    X = embs[:, 0]
    Y = embs[:, 1]
    labels = names

    for (id, x, y, label, assembly, internal_id) in zip(
        ids,
        X,
        Y,
        labels,
        assembly_code,
        internal_ids
    ):
        rows.append(
            {
                "id": id,
                "label": label,
                "x": float(x) * 100,
                "y": float(y) * 100,
                "fixed": True,
                "custom_id": internal_id,
            },
        )
        with open(f"cached/{internal_id}", "w") as file:
            file.write(str(assembly))
    return rows

if __name__ == "__main__":
    if os.path.isfile("dump.json"):
        with open("dump.json", "r") as file:
            rows = json.load(file)
    print(get_cluster_count(rows))
