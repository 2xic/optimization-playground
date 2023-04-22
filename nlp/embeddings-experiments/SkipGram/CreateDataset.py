from .Parameters import OUTPUT_SIZE
import random

class CreateDataset:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        """
        SkipGram -> 
            Context word should get higher score
        """

    def process(self, documents):
        X = []
        y = []
        for doc in documents:
            for index, token in enumerate(doc):
                print(token, end=" ")
                if token < self.vocab.SPECIAL_TOKENS:
                    continue
                middle = OUTPUT_SIZE // 2
                valid_context = []
                for backward_context_idx in range(max(0, index - middle), index):
                    valid_context.append(
                        doc[backward_context_idx]
                    )
                for forward_context_idx in range(index, min(len(doc), index + middle)):
                    valid_context.append(
                        doc[forward_context_idx]
                    )
                filtered = list(filter(lambda x: x > self.vocab.SPECIAL_TOKENS, valid_context))
                if len(filtered):
                    X.append(token)
                    y.append(filtered)#[random.randint(0, len(filtered) - 1)])
        return X, y
