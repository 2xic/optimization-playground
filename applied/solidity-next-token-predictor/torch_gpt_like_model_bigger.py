import os
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from pre_generator import get_cache_file
import atexit
import time
from optimization_playground_shared.utils.General import save_model_atomic, load_model, does_model_exists
from torch.cuda.amp import autocast
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import json
import optimization_playground_shared
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset

metrics_tracker = Tracker("solidity_next_token_predictor").send_code_state([
    __file__,
    optimization_playground_shared.__file__
]) if __name__ == "__main__" else None

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 8
# Need to try to tain model with long sequence size ....
SEQUENCE_LENGTH = 256
CACHE_FILE = ".model_state.pkt"

def get_model(vocab):
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=8,
        dropout=0.1,
        sequence_length=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
        transformer_layers=8,
        attention_heads=4,
        feed_forward=256,
    )
    model = GptTransformerModel(config)
    return model

def get_cached_model(vocab):
    vocab.lock()
    model = get_model(vocab)
    # TODO: fix this
    if does_model_exists(CACHE_FILE):
        print("Loading cached model")
        checkpoint = load_model(CACHE_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def flatten_view(x, y):
    return x, y.view(-1)

def train_loop(vocab: SimpleVocab, model: GptTransformerModel):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
#    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss())#ignore_index=vocab.vocab.PADDING_IDX))
    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))

    dataloader = ZmqDataloader()
    iterator = iter(dataloader)
    last_save = 0

    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
#    model.embedding.cuda(0)
#    model.pos_encoder.cuda(0)
#    model.transformer_decoder.cuda(1)
    # Output should match device of the dataloader ... 
#    model.output.cuda(2)
    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
    print("")

    for epoch in range(1024):
        X, y = get_document_dataset(vocab, [
            next(iterator) for _ in range(32)
        ])
        raw_dataloader = get_raw_dataloader((
            X.clone(),
            y.clone()
        ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        
        with autocast():
            (loss, accuracy) = trainer.use_tqdm().train(
                raw_dataloader,
                callback=flatten_view,
            ) #, sharding=True)
        # Try to predict something 
        stats = {}
        for x in X.tolist():
            for token in x:
                if token not in stats:
                    stats[token] =1
                else:
                    stats[token] += 1
        metric = Metrics(
            epoch=epoch,
            loss=loss.item(),
            training_accuracy=accuracy.item(),
            prediction=Prediction.text_prediction(
                "\n".join([
                    "Example 1:",
                    vocab.decode(
                        model.rollout(
                            vocab.get_tensor("uint8 ", sequence_length=-1)[0],
                            steps=100,
                            sampling="argmax"
                        )
                    ),
                    json.dumps(
                        vocab.get_tensor("uint8 ", sequence_length=-1).tolist(),
                    ),
                    json.dumps(model.rollout(
                        vocab.get_tensor("uint8 ", sequence_length=-1)[0],
                        steps=100,
                        sampling="argmax"
                    )),
                    "Example 2",
                    vocab.decode(
                        model.rollout(
                            vocab.get_tensor("contract Uniswap {", sequence_length=-1)[0],
                            steps=100,
                            sampling="temperature"
                        )
                    ),
                    "Example 3:",
                    vocab.decode(
                        model.rollout(
                            vocab.get_tensor("contract ", sequence_length=-1)[0],
                            steps=100,
                            sampling="argmax"
                        )
                    ),
                    "Example 4:",
                    vocab.decode(
                        model.rollout(
                            vocab.get_tensor("// This contract is part", sequence_length=-1)[0],
                            steps=100,
                            sampling="argmax"
                        )
                    ),
                    "\n\n\n\n",
                    "X[0]",
                    vocab.decode(
                        X[0].tolist()
                    ),
                    "stats for X",
                    json.dumps(stats),
                #    "y[0]",
                #    vocab.decode(
                #        [y[0].tolist()]
                #    )
                ])
            )
        )
        metrics_tracker.log(metric)

        if (time.time() - last_save) > 60:
            save_model_atomic(CACHE_FILE, model)
            last_save = time.time()
    return model

if __name__ == "__main__":
    def save_model():
        print("STARTING TO SAVE MODEL")
    #    save_model_atomic(CACHE_FILE, model)
        print("DONE SAVING MODEL")

    atexit.register(save_model)

    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    assert vocab is not None
    print("Loaded vocab")
    model = get_cached_model(vocab)
    print("Loaded model")
    model = train_loop(vocab, model)
    print(model)
