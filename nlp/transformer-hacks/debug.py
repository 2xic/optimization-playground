from datasets.web.web_dataset import WebDataset
from datasets.bytecode.bytecode_dataset import BytecodeDatasetTiny
import time

# Doesn't support torch
# import platform
# impl_name = platform.python_implementation()
# assert impl_name == "PyPy"

# dataset = WebDataset()
# dataset.create_tokenizer()

# Baseline
# 2.039179563522339
# 2.0622174739837646
# 2.053544759750366
start = time.time()
# tok, dataset = dataset.create_dataset()  # recreate=True)

# for X, y in dataset.iter():
#    print(tok.decode(X[0].tolist()))

BytecodeDatasetTiny().create_dataset(sequence_size=512, recreate=True)

end = time.time()
print(end - start)
