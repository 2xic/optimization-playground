from dotenv import load_dotenv
load_dotenv()
from embeddings import HuggingFaceWrapper
import numpy as np
import time


if __name__ == "__main__":
    # model = "meta-llama/Meta-Llama-3.1-8B"
    # model = "meta-llama/Meta-Llama-3-8B"
    # model = "meta-llama/Llama-2-7b-hf"
#    model = "intfloat/multilingual-e5-large"
    model = "nvidia/NV-Embed-v1"
    start = time.time()
    output = HuggingFaceWrapper(
        model=model
    ).transforms([
        "test"
    ])
    end = time.time()
    print(output)
    print(np.asarray(output[0]).shape)
    print(model)
    print(end - start)
