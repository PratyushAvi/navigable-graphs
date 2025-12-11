import struct
import numpy as np
import os

BINARY_PATH = "/scratch/pa2439/Projects/ANN-Search/BILLIONDATASET/SPTAG/datasets/SPACEV1B/vectors.bin"
SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B"

# Read metadata from first file
with open(os.path.join(BINARY_PATH, 'vectors_1.bin'), 'rb') as f:
    vec_count = struct.unpack('i', f.read(4))[0]
    vec_dimension = struct.unpack('i', f.read(4))[0]

print(f"Vector count: {vec_count}, dimension: {vec_dimension}")
print(f"Total size: {vec_count * vec_dimension / 1e9:.2f} GB")

# Use direct file I/O instead of memmap writes
part_count = len(os.listdir(BINARY_PATH))

with open(f'{SAVEPATH}/vectors_int.mmap', 'wb') as out_file:
    for i in range(1, part_count + 1):
        print(f"Processing part {i}/{part_count}...")
        
        with open(os.path.join(BINARY_PATH, f'vectors_{i}.bin'), 'rb') as f:
            if i == 1:
                # Skip header in first file only
                f.read(8)
            
            # Copy in 100MB chunks
            while True:
                chunk = f.read(100 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

print("Conversion complete!")

# Save metadata
# np.save(f'{SAVEPATH}/vectors_shape.npy', np.array([vec_count, vec_dimension]))

# Verify
X = np.memmap(f'{SAVEPATH}/vectors_int.mmap', dtype='int8', mode='r', 
              shape=(vec_count, vec_dimension))
print(f"Verified: shape={X.shape}, dtype={X.dtype}")