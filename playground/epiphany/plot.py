import matplotlib.pyplot as plt
import json

data = None
with open("results.json", "r") as file:
    data = json.load(file)

plt.plot(data["training"])
plt.savefig('training.png')
plt.clf()

plt.plot(data["testing"])
plt.savefig('testing.png')
