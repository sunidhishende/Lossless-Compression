import os
import subprocess
from datasets import load_dataset, DatasetDict

archive_name = 'temp.zpaq'
output_dir = 'extracted_files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

command = f'zpaq x {archive_name} -to {output_dir}'

print(f'Running command: {command}')
result = subprocess.run(command, shell=True)

if result.returncode == 0:
    print("Extraction complete.")
else:
    print("Error extracting files.")

splits = ['0', '1']
ds = load_dataset('commaai/commavq', num_proc=40, split=splits)
ds = DatasetDict(zip(splits, ds))

