"""
We know the overall score, we want to find a neuron that seems correlated to this value

How we can approach this
1. Iterate over each neuron and store it in a dictionary
2. Based on its firing we create a average score over the entire text
3. This is then added to a global counter
4. Then look for a neuron which seems to consistent have a good matching value to the review score

"""
from model import Model
from model import Model
from vocab import Vocab
from dataset import Dataset
import torch
import matplotlib.pyplot as plt

max_size = 400
    
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



def look_for_sentiment_neuron(model: Model, vocab: Vocab, device):
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
    
    text, sentiment = Dataset(dataset_entries=100).get_score_and_text(max_size) 
    text, sentiment = zip(*list(sorted(list(zip(text, sentiment)), key=lambda x: x[1])))
    for tokens in text:
        _ = model.forward_feed(vocab.get_encoded(tokens, device=device))
        # create one neuron layer map
        for output_values in [
            saved_output_1,
            saved_output_2,
            saved_output_3,
            saved_output_4,
        ]:
            response = output_values.get_and_clear()
            scores = {}
            for i in response:
                for index, neuron_value in enumerate(i.reshape((-1)).tolist()):
                    if index in scores:
                        scores[index].append(neuron_value)
                    else:
                        scores[index] = [neuron_value]
            for i in list(scores.keys()):
                scores[i] = sum(scores[i]) / len(scores[i])
                # this will then later be shown so we can match it
                output_values.add_firing_value(i, scores[i])
    # index -> [best neuron index, score]
    best_layer_score_index = {}
    best_neuron_firing_scores = {}
    scores = sentiment
    for index, i in enumerate([
            saved_output_1,
            saved_output_2,
            saved_output_3,
            saved_output_4,
        ]):
        neuron_score = []
        for neuron_id, value in i.firing_scores.items():
            error = ((torch.tensor(value) - torch.tensor(scores)) ** 2).sum().item()
            neuron_score.append([neuron_id, error])
        best_score = sorted(neuron_score, key=lambda x: x[1])[0]
        best_layer_score_index[index] = best_score
        # best_neuron_id
        best_neuron_id = best_score[0]
        best_neuron_firing_scores[f"{index}_{best_neuron_id}"] = i.firing_scores[best_neuron_id]
    print(best_layer_score_index)
    print(best_neuron_firing_scores)
    plt.plot(list(range(len(scores))), scores, label="sentiment")
    for key, value in best_neuron_firing_scores.items():
        print((max(value), min(value), key))
        plt.plot(list(range(len(value))), value, label=key)
    plt.legend(loc="upper left")
    plt.savefig(f'sentiment_neuron.png')

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = Vocab().load()
    model = Model.load().to(device)
    look_for_sentiment_neuron(model, dataset, device)
