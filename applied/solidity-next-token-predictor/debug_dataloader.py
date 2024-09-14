from torch_gpt_like_model_bigger import get_document_dataset
from pre_generator import get_cache_file
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader

if __name__ == "__main__":
    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    dataloader = ZmqDataloader()
    for document in dataloader:
        X, y = get_document_dataset(vocab, [document])
        # print(y)
        a = y.reshape((-1)).tolist()
   #     print(vocab.decoded_tokens(a))
        stats = {}
        for index, item in enumerate(X.tolist()):
            for token in item:
                if token not in stats:
                    stats[token] =1
                else:
                    stats[token] += 1
#        print(stats)
#        print(vocab.decoded_tokens([0]))
        print(stats[0])

        break
