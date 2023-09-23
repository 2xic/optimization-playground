import unittest
from model import Model, Config
from vocab import Vocab

class TestAgent(unittest.TestCase):
    def test_forward(self):
        dataset = Vocab()
        x, y = dataset.get_dataset([
            "hello world, I hope you are doing well today",
            "this is some text about text",
            "excellent! wonderful"
        ])
        assert len(x.shape) == 2
        config = Config(
            tokens=dataset.get_vocab_size(),
            padding_index=dataset.PADDING_IDX,
            sequence_size=1
        )
        model = Model(config)
        loss = model.fit(x, y)
        assert loss is not None
    
    
