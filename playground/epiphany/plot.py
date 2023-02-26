import matplotlib.pyplot as plt
import json

data = None
with open("results.json", "r") as file:
    data = json.load(file)

plt.title('Accuracy training dataset on CIFAR-10')
plt.plot(data["x"], data["training"])
plt.xscale('symlog')
plt.savefig('training.png')
plt.clf()

plt.title('Accuracy on test dataset on CIFAR-10')
plt.plot(data["x"], data["testing"])
plt.xscale('symlog')
plt.savefig('testing.png')
