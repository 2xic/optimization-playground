import os
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from pre_generator import get_cache_file
from optimization_playground_shared.utils.General import save_model_atomic, load_model, does_model_exists
from torch.cuda.amp import autocast
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import optimization_playground_shared
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.dataloaders.Utils import find_batch_size
import atexit
import time
import json

metrics_tracker = Tracker("solidity_next_token_predictor").send_code_state([
    __file__,
    optimization_playground_shared.__file__
]) if __name__ == "__main__" else None

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 1
PRELOAD_VOCAB = False
# Need to try to tain model with long sequence size ....
SEQUENCE_LENGTH = 256
CACHE_FILE = ".model_state.pkt"
DOCUMENTS_TO_LOAD_PER_BATCH = 1

def get_model(vocab):
    embedding_dim = 32
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=embedding_dim,
        dropout=0,
        sequence_length=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
        transformer_layers=8,
        attention_heads=4,
        feed_forward=embedding_dim * 4,
    )
    model = GptTransformerModel(config)
    return model

def get_cached_model(vocab: SimpleVocab):
    vocab.lock()
    model = get_model(vocab)
    # TODO: fix this
    #if does_model_exists(CACHE_FILE):
    #    #print("Loading cached model")
    #    #checkpoint = load_model(CACHE_FILE)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def flatten_view(x, y):
    return x, y.view(-1)

def _accuracy_check(predicted, y, vocab: SimpleVocab):
    predicted = torch.argmax(
        predicted,
        dim=1
    )

    print(predicted)
    print(y)

    indices = torch.where(y != vocab.vocab.PADDING_IDX)
    y_filtered = y[indices]
    accuracy = ((y[indices] == predicted[indices]).sum())

    return accuracy, y_filtered.shape[0]

def train_loop(vocab: SimpleVocab):
    dataloader = ZmqDataloader()
    iterator = iter(dataloader)
    X, y = get_document_dataset(vocab, [
        next(iterator) for _ in range(DOCUMENTS_TO_LOAD_PER_BATCH)
    ], SEQUENCE_LENGTH=SEQUENCE_LENGTH)
    raw_dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=1,
        shuffle=True,
    )
    model = get_cached_model(vocab)
    print("Loaded model")

    optimizer = optim.Adam(model.parameters())
#    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss())#ignore_index=vocab.vocab.PADDING_IDX))
    trainer = TrainingLoop(
        model, 
        optimizer, 
        loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX),
        callback=flatten_view,
    )
    raw_dataloader = find_batch_size(trainer, raw_dataloader, device=torch.device('cuda:0'))
    # print(vocab.size)

    trainer._accuracy_check = lambda x, y: _accuracy_check(x, y, vocab)

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
                        model.rollout_output(
                            vocab.encode("uint8 "),
                            steps=100,
                            sampling="argmax"
                        )
                    ),
                    json.dumps(
                        vocab.encode("uint8 "),
                    ),
                    json.dumps(model.rollout_output(
                        vocab.encode("uint8 "),
                        steps=100,
                        sampling="argmax"
                    )),
                    "Example 2",
                    vocab.decode(
                        model.rollout_output(
                            vocab.encode("contract Uniswap {"),
                            steps=100,
                            sampling="temperature"
                        )
                    ),
                    "Example 3:",
                    vocab.decode(
                        model.rollout_output(
                            vocab.encode("contract "),
                            steps=100,
                            sampling="argmax"
                        )
                    ),
                    "Example 4:",
                    vocab.decode(
                        model.rollout_output(
                            vocab.encode("// This contract is part"),
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
    vocab = get_cache_file() if PRELOAD_VOCAB else SimpleVocab()
    assert vocab is not None
    print("Loaded vocab")
    model = train_loop(vocab)
    print(model)
