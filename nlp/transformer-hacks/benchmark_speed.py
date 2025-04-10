from transformer_dataset import PartialMemoryTensor, TransformerTextDatasetLazy
from dataset_tokenizer import WordPiece
from train import train,TrainingOptions
from tqdm import tqdm

"""
create_iterator = glob.iglob(
    "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol",
    recursive=True
)
N = 2_000
#N = 1_00

items = [
    next(create_iterator) for i in range(N)
]
N_ITEMS = 71_698

name = "smart_contract_fiesta"
(new_tokenizer, cached) = WordPiece.load_cache(name)

start = time.time()
print("Starting test")

#TransformerTextDataset.from_iterator("benchmark", new_tokenizer, items, 256)
TransformerTextDataset.from_iterator_single("benchmark", new_tokenizer, items, 256)

print((time.time() - start) / N)
print("Total time:")
print((time.time() - start) / N * (N_ITEMS / N))
"""




name = "smart_contract_fiesta"
(new_tokenizer, cached) = WordPiece.load_cache(name)

text_dataset = TransformerTextDatasetLazy("smart_contract_fiesta", new_tokenizer)

# print(dataset[20])
# print(len(dataset))
"""
ids = PartialMemoryTensor("smart_contract_fiesta").ids()
for (id, _) in ids:
    X, y = PartialMemoryTensor("smart_contract_fiesta").load(id)

    print((X[0]))
    print((y[0]))
"""

train(
    text_dataset,
    options=TrainingOptions(
        batch_size=256
    ),
    progress=lambda x: tqdm(range(x))
)
