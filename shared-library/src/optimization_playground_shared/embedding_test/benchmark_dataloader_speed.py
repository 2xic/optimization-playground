""""
Loading the dataset should not be a bottleneck. We need to be able to run the dataset 
fast. 
"""
# python3 -m optimization_playground_shared.embedding_test.benchmark_dataloader_speed

from .dataloader import get_dataloader
from .evals import EvaluationMetrics
from ..nlp.DocumentEncoderSequence import SimpleVocab
import time
document_encoder: SimpleVocab = SimpleVocab().load(
    "/root/", 
    prefix="pretrained"
)
print(len(document_encoder.vocab.index_vocab))

#for i in get_dataloader()
dataloader, dataset = get_dataloader(
    document_encoder, 
    "next_token_prediction",
    8,
)
"""
delta = time.time()
for (X, y) in dataloader:
    new_time = time.time()
    print(X.shape, new_time - delta)
    delta = new_time
"""
evals = EvaluationMetrics()
print(len(evals.X_test_original))


