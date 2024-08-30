import numpy as np
from datasets import load_dataset, DatasetDict
import os
import lzma
import zlib
import bz2
import zstandard
import multiprocessing
# from bitsandbytes import huffman
import shutil
import pickle
from pathlib import Path

output_dir = Path('./temp/')
num_proc = multiprocessing.cpu_count() 
base_path = 'commavq/data_0_to_5000/'

def compress_tokens(tokens: np.ndarray) -> bytes:
  tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes() # transposing increases compression rate ;)
#   tokens = tokens.astype(np.int16).reshape(-1, 128).ravel().tobytes()
  return lzma.compress(tokens)

def compress_example(example):
  path = Path(base_path+example['path'])
  tokens = np.load(path)
  compressed = compress_tokens(tokens)
  compression_rate = (tokens.size * 10 / 8) / len(compressed) # 10 bits per token
  with open(output_dir/path.name, 'wb') as f:
    f.write(compressed)
  example['compression_rate'] = compression_rate
  return example

if __name__ == '__main__':
    splits = ['0', '1']
    ds_filename = 'dataset.pkl'
    if not os.path.exists(ds_filename):
        ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits)
        ds = DatasetDict(zip(splits, ds))
        with open(ds_filename, 'wb') as f:
            pickle.dump(ds, f)
    else:
        with open(ds_filename, 'rb') as f:
            ds = pickle.load(f)
    os.makedirs(output_dir, exist_ok=True)
    ratios = ds.map(compress_example, desc="compress_example", num_proc=num_proc, load_from_cache_file=False)
    # make archive
    #   shutil.copy('./compression/decompress.py', output_dir)
    shutil.make_archive('temp', 'zip', output_dir)
    # print compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize("temp.zip")
    print(f"Compression rate: {rate:.1f}")