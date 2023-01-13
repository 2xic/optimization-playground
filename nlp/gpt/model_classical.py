import torch.nn as nn
import torch
from vocab import Vocab
from ClassicalAttention import ClassicalAttentionLayer
from train_loop import train_loop
from optimization_utils.logging.EpochRuns import EpochRuns
from MutltiHeadAttention import MutltiHeadAttention
from classical_attention_alt import Attention

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")

class Model(nn.Module):
    def __init__(self, vocab_size, USE_ATTENTION):
        super().__init__()

        self.SEQ_SIZE = 8
        position_size = 64
        self.vocab_size = vocab_size
        self.embedding_dim = 32

#        USE_ATTENTION = False

        self.ATTENTION_SHAPE = 1024 // 3
        self.LINEAR_SHAPE = self.embedding_dim * self.SEQ_SIZE

        self.word_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim=self.embedding_dim,
         #   padding_idx=-1
        )
        self.position_embedding = nn.Embedding(
            position_size, 
            embedding_dim=self.embedding_dim,
        #    padding_idx=-1
        )

        self.block_converter = lambda x: x# nn.Linear(self.LINEAR_SHAPE, self.ATTENTION_SHAPE)

        self.blocks = torch.nn.Sequential(*(
                [
                    nn.Sigmoid(),
                    Attention(self.embedding_dim),
                    #MutltiHeadAttention(self.ATTENTION_SHAPE, self.ATTENTION_SHAPE),                    
                ]    
                if USE_ATTENTION 
                else [
                    nn.Sigmoid()
                ]
            )
        )
        self.lm_head = nn.Linear(self.embedding_dim * self.SEQ_SIZE, vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.epoch_information = EpochRuns(
            "classical_attention" if USE_ATTENTION else "non_attention_linear"
        )


    def forward(self, text, position):
        #print("shape ", text.shape)
        h_0 = self.word_embedding(text) + self.position_embedding(position)
     #   h_0 = h_0.reshape((h_0.shape[0], self.embedding_dim * self.SEQ_SIZE))
     #   h_0 = torch.sigmoid(self.block_converter(h_0))

        h_1 = self.blocks(h_0).reshape((h_0.shape[0], self.embedding_dim * self.SEQ_SIZE))
      #  print(h_0.shape)
     #   print(h_1.shape)
    #    exit(0)
   #     h_1 = torch.sigmoid(h_1)
   #     print(text.shape)
  #      print(h_1.shape)
 #       print(self)
#        print(self.lm_head(h_1))
        return self.log_softmax(self.lm_head(h_1))
#        return self.log_softmax(self.lm_head(h_1.reshape((text.shape[0], self.ATTENTION_SHAPE))))


if __name__ == "__main__":
    vocab = Vocab()
    X = vocab.encode_file("large_tet_file")
#    print(X.shape)

    model = Model(
        vocab.lock(),
        USE_ATTENTION=True
    )
    optimizer = torch.optim.Adam(model.parameters())
    count_parameters(model)
#    exit(0)

    train_loop(
        model=model,
        vocab=vocab,
        X=X,
        optimizer=optimizer,
        epochs=2_000
    )
