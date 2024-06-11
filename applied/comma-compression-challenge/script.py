# MODIFIED from https://github.com/commaai/commavq/blob/master/compression/compress.py
import os
import lzma
import zlib
import brotli
import multiprocessing
import shutil
import numpy as np
import glob
import bz2
import shutil
import snappy

from pathlib import Path
from datasets import load_dataset, DatasetDict, DownloadManager, DownloadConfig

compression = "bz2"
libraries = {
  "lzma": {
    "compress": lzma.compress,
    "decompress": lzma.decompress,
  }, # 1.6
  "zlib": {
    "compress": zlib.compress,
    "decompress": zlib.decompress,
  }, # 1.3
  "brotli": {
    "compress": brotli.compress,
    "decompress": brotli.decompress,
  }, # 1.5
  "snappy": {
    "compress": snappy.compress,
    "decompress": snappy.decompress,
  }, # 1.1
  "bz2": {
    "compress": bz2.compress,
    "decompress": bz2.decompress,
  }, # 1.5
}

output_dir = Path('./compression_challenge_submission/')
data_dir = Path("./data_dir/")

def compress_tokens(tokens: np.ndarray) -> bytes:
  tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes() # transposing increases compression rate ;)
  return libraries[compression]["compress"](tokens)

def decompress_bytes(x: bytes) -> np.ndarray:
  tokens = np.frombuffer(libraries[compression]["decompress"](x), dtype=np.int16)
  return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def compress_example(example):
  path = Path(example['path'])
 # print(path)
  tokens = np.load(data_dir.joinpath(path))
  compressed = compress_tokens(tokens)
  compression_rate = (tokens.size * 10 / 8) / len(compressed) # 10 bits per token
  with open(output_dir/path.name, 'wb') as f:
    f.write(compressed)
  assert np.all(tokens == decompress_bytes(compressed)), f"decompressed data does not match original data for {path}"
  example['compression_rate'] = compression_rate
  return example

def setup():
  for i in ["data_0_to_2500", "data_2500_to_5000"]:
    DownloadManager('commaai/commavq', download_config=DownloadConfig(
      cache_dir=os.path.join(
      os.path.dirname(__file__),
      "data_raw_dir"
    ))).download_and_extract(f"https://huggingface.co/datasets/commaai/commavq/resolve/main/{i}.zip")
    for i in glob.glob("data_raw_dir/**/*.npy", recursive=True):
        if not os.path.isfile(os.path.join(data_dir, os.path.basename(i))):
          shutil.copy(i, data_dir)

if __name__ == '__main__':
#  setup()
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)
  num_proc = multiprocessing.cpu_count()
  # load split 0 and 1
  splits = ['0', '1']
  ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits)
  ds = DatasetDict(zip(splits, ds))
  ratios_zip = {}
  ratios_tar = {}
  for names in libraries.keys():
    compression = names
    ratios = ds.map(compress_example, desc="compress_example", num_proc=num_proc, load_from_cache_file=False)
    # make archive
    shutil.make_archive('compression_challenge_submission', 'zip', output_dir)
    # print compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize("compression_challenge_submission.zip")
    print(f"Compression rate ({compression}.zip): {rate:.1f}")
    ratios_zip[compression] = f"{rate:.1f}"
    shutil.make_archive('compression_challenge_submission', 'tar', output_dir)
    # print compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize("compression_challenge_submission.tar")
    print(f"Compression rate ({compression}.tar): {rate:.1f}")
    ratios_tar[compression] = f"{rate:.1f}"
print(ratios_zip)
print(ratios_tar)
