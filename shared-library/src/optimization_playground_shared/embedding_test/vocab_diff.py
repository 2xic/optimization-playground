"""
python3 -m optimization_playground_shared.embedding_test.vocab_diff
"""

from ..nlp.DocumentEncoderSequence import SimpleVocab
from ..nlp.wordpiece.bpe import BPE

vocab = SimpleVocab().load(".", prefix="pretrained")
print(vocab.size)

vocab = BPE().load(".", prefix="pretrained")
print(vocab.size)
