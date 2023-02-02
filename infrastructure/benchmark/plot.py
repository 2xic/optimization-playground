import numpy as np
import matplotlib.pyplot as plt
import json

bar_width = 0.25
fig = plt.subplots(figsize =(12, 8))


versions = [
    "torch_1.13.0+cu117",
    "torch_2.0.0.dev20230202+cu117"
]
models = {

}
for i in versions:
    data = None
    with open(f"{i}.json", "r") as file:
        data = json.load(file)
    for key, value in data.items():
        if key not in models:
            models[key] = {}
            models[key][i] = {}
        elif i not in models[key]:
            models[key][i] = {}
        models[key][i] = value

len_version = len(versions)
len_models = len(models)

models_linear = {}

for i in models:
    for v in versions:
        for is_jit in ["jit", "non_jit"]:
            key = f"{i}/{is_jit}"
            if key not in models_linear:
                models_linear[key] = []
            models_linear[key].append(models[i][v][is_jit])

bar_width = 0.25
fig = plt.subplots(figsize =(12, 8))
 
colors = ['b', 'g', 'orange', 'yellow']
names = list(models_linear.keys())
models = sum([
    [
        models_linear[f"{key}/jit"],
        models_linear[f"{key}/non_jit"
    ]] 
    for key in models
], [])


for index, value in enumerate(models):
    x_location = [
        c_index * len(names) + bar_width * index
        for c_index, _ in enumerate(range(len(versions)))
    ]
    plt.bar(
        x_location, 
        value, 
        color=colors[index], 
        width = bar_width, 
        edgecolor ='grey', 
        label = names[index]
    )
 
plt.xlabel('Versions', fontweight ='bold', fontsize = 15)
plt.ylabel('Speed (ns)', fontweight ='bold', fontsize = 15)
plt.xticks([0, 4], versions)
 
plt.legend()
plt.savefig('plot.png')

