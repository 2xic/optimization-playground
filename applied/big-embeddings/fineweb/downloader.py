"""
https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/CC-MAIN-2013-20?layout=container&viewer_api=true

You can download individual parque files

https://huggingface.co/docs/dataset-viewer/parquet
"""
import requests
import os
from datasets import load_dataset

def get_dataset():
    output_file = "0.parquet"
    if not os.path.isfile(output_file):
        url = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/parquet/CC-MAIN-2013-20/train/0.parquet"
        response = requests.get(url, stream=True)

        file_bytes = bytearray()

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file_bytes.extend(chunk)
            print(f"Downloaded {len(file_bytes)} bytes")
            with open(output_file, "wb") as file:
                file.write(file_bytes)
        else:
            print("Failed to download file")

    dataset = load_dataset("parquet", data_files={'train': output_file})
    for _, dataset in dataset.items():
        for example in dataset:
            yield example["text"]

if __name__ == "__main__":
    get_dataset()
