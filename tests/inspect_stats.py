import numpy as np
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'test_stats.npz'

data = np.load(filename)
print(f'\nStatistics file: {filename}')
print('='*80)
for key in sorted(data.keys()):
    arr = data[key]
    if hasattr(arr, 'shape'):
        print(f'{key:20s}: shape={str(arr.shape):20s} min={arr.min():12.6e} max={arr.max():12.6e}')
    else:
        print(f'{key:20s}: {arr}')
