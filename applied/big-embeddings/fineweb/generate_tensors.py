from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from tqdm import tqdm
import torch 
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
from downloader import get_dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter

def generate():
    bpe = BPE(
        show_progress=False
    )
    # Open the zip file
    text_content = []
    
    for content in get_dataset():
        tokens = splitter(content)
        bpe.add_tokens(tokens)
        text_content.append(content)

        if len(text_content) > 128:
            break

    start = time.time()
    minutes = 5
    seconds = minutes * 60
    progress_interval = 10
    ten_sends = minutes * 60 // progress_interval
    with tqdm(total=ten_sends) as pbar:
        previous = time.time()
        while (time.time() - start) < seconds:
            bpe.merge()
            if time.time() - previous > progress_interval:
                pbar.update(1)
                previous = time.time()

    with open("bpe", "wb") as file:
        pickle.dump(bpe, file)

    with ThreadPoolExecutor(max_workers=16) as executor:
        tensors = []
        for text in tqdm(text_content):
            tensors.append(executor.submit(bpe.encode, text))
        for index, v in enumerate(tensors):
            tensors[index] = v.result()
        torch.save(tensors, 'tensors.pt')

if __name__ == "__main__":
    generate()

