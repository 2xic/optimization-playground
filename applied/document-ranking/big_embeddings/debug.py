from .big_gpt import get_model, SEQUENCE_LENGTH
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset


bpe, _ = get_model()

sequence = [
    "hello world, I love bagels. Maybe if I try to overfit the model then this thing will work."
    for _ in range(2)
]
X, y = get_document_dataset(bpe, sequence, SEQUENCE_LENGTH)

for index in range(X.shape[0]):
    print(bpe.decode(X[index].tolist()))
    print(bpe.decode(y[index].tolist()))

