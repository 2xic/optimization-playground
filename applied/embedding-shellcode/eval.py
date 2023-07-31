"""
Eval script to make sure the output is reasonable
"""

import os
import json
from optimization_playground_shared.clustering.elbow_method import ElbowMethod
import os
from sklearn.manifold import TSNE
import json
from get_shellcode import get_shellcode

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
    rows = []
    code = list(get_shellcode())
    names = list(map(lambda x: x[1], code))
    assembly_code = list(map(lambda x: x[0], code))
#    print(names)

    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    embs = tsne.fit_transform(prediction)

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
#        print((x, y, label, id))
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
    return rows

if __name__ == "__main__":
    if os.path.isfile("dump.json"):
        with open("dump.json", "r") as file:
            rows = json.load(file)
    print(get_cluster_count(rows))
