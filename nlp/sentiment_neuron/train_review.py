from model import Model, Config
from vocab import Vocab
import torch.optim as optim
from dataset import Dataset
import torch
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction

metrics_tracker = Tracker("review_predictions")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Vocab()
max_document_length = 400 # number of chars
review_to_learn_from = 10_000
sentences = Dataset(dataset_entries=review_to_learn_from).get_text(max_document_length)
dataset = dataset.process_dataset(sentences).lock()
dataset.save()
    
config = Config(
    tokens=dataset.get_vocab_size(),
    padding_index=dataset.PADDING_IDX,
    sequence_size=1,
    embedding_dim=16,
)
print(config)

model = Model(config).to(device)
print("decoded=",dataset.decode([2]))
batch_size = 32
optimizer = optim.Adam(model.parameters())
print("Starting to train ...")
"""
It is a bit slow to start with, but it learns after a few 100 epochs
"""
for epoch in range(10_000):
    loss = 0
    latest_loss = None
    for index in range(0, len(sentences), batch_size):
        x, y = dataset.get_dataset(sentences[index:index + batch_size], device=device)
        x_decoded, y_decoded = dataset.decode(x.reshape(-1).tolist()), dataset.decode(y.reshape(-1).tolist())
        loss += model.fit(x, y, debug=True)

        if index % batch_size * 10 == 0:
            metric = Metrics(
                epoch=epoch,
                loss=loss.item(),
                training_accuracy=None,
                prediction=None
            )
            metrics_tracker._log(metric)
            latest_loss = loss
            print(f"loss {loss} epoch: {epoch}, epoch progress {index} / {len(sentences)}")
            # batching together to get more epoch like flow
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

    results = []
    for chunk_sizes in [4, 8, 16, 32]:
        seed = x_decoded[:chunk_sizes]
        predict_vector = dataset.get_encoded(seed, device=device)
        output = model.predict(predict_vector, debug=False)
        predicted = dataset.decode(output)
        truth = x_decoded[:max_document_length]
        results.append(f"Seed: {seed}\n\nTruth: {truth} \n\nPredicted rollout: {predicted}")
    metric = Metrics(
        epoch=epoch,
        loss=latest_loss,
        training_accuracy=None,
        prediction=Prediction.text_prediction(
            "\n\n".join(results)
        )
    )
    metrics_tracker._log(metric)

    print("predicted", predicted)
    print("truth", truth)
    print("")
    model.save()
    
