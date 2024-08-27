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
        print(vocab.decoded_tokens(a))
