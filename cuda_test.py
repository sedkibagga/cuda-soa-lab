# Minimal CUDA kernel test for lab setup
from numba import cuda
import numpy as np

@cuda.jit
def add_one_kernel(x, out):
	idx = cuda.grid(1)
	if idx < x.size:
		out[idx] = x[idx] + 1

def test_cuda():
	a = np.arange(10, dtype=np.float32)
	d_a = cuda.to_device(a)
	d_out = cuda.device_array_like(d_a)
	add_one_kernel[2, 5](d_a, d_out)  # 2 blocks, 5 threads per block
	out = d_out.copy_to_host()
	print("Input:", a)
	print("Output:", out)

if __name__ == "__main__":
	test_cuda()
