# ------------------------------------
# How CUDA Kernels Work (Numba in Python)
# ------------------------------------
"""
A CUDA kernel is a function that runs on the GPU and is executed in parallel by many threads. In Python, using Numba's @cuda.jit decorator, you can write such kernels that are compiled for the GPU. Each thread executes the same code but on different data, allowing massive parallelism.

Key concepts:
- The kernel is decorated with @cuda.jit and launched from the host (CPU) code.
- Data must be explicitly transferred from host (CPU) to device (GPU) using cuda.to_device(...).
- After kernel execution, results are copied back to the host with copy_to_host().
- Threads are grouped into blocks, and blocks form a grid. You can use cuda.grid(2) for 2D indexing.
- Example minimal kernel:

	from numba import cuda
	import numpy as np

	@cuda.jit
	def add_gpu(x, out):
		idx = cuda.grid(1)
		out[idx] = x[idx] + 2

	a = np.arange(10, dtype=np.float32)
	d_a = cuda.to_device(a)
	d_out = cuda.device_array_like(d_a)
	add_gpu[2, 5](d_a, d_out)  # 2 blocks, 5 threads per block
	out = d_out.copy_to_host()
	print(out)

This approach allows for high-performance computation on large data, provided the problem size is sufficient to amortize data transfer overheads.
"""
# FastAPI GPU Matrix Addition Service
# ------------------------------------
# This service exposes endpoints for matrix addition on GPU, health check, and GPU info.
# It uses Numba's CUDA JIT to accelerate matrix addition on NVIDIA GPUs.
#
# Endpoints:
#   POST /add      - Upload two .npz files, add matrices on GPU, return shape, time, device
#   GET  /health   - Health check
#   GET  /gpu-info - Show GPU memory usage (via nvidia-smi)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from numba import cuda
import time
import subprocess

app = FastAPI()

# CUDA kernel for matrix addition
@cuda.jit
def matadd_kernel(A, B, C):
	i, j = cuda.grid(2)
	if i < A.shape[0] and j < A.shape[1]:
		C[i, j] = A[i, j] + B[i, j]

def gpu_matrix_add(A: np.ndarray, B: np.ndarray):
	# Allocate device memory
	d_A = cuda.to_device(A)
	d_B = cuda.to_device(B)
	d_C = cuda.device_array_like(A)
	# Configure blocks and grids
	threadsperblock = (16, 16)
	blockspergrid_x = (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
	blockspergrid_y = (A.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	# Launch kernel and time
	start = time.perf_counter()
	matadd_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)
	cuda.synchronize()
	elapsed = time.perf_counter() - start
	# Copy result back
	C = d_C.copy_to_host()
	return C, elapsed

@app.post("/add")
async def add_matrices(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
	try:
		a_bytes = await file_a.read()
		b_bytes = await file_b.read()
		A = np.load(a_bytes)[list(np.load(a_bytes).files)[0]]
		B = np.load(b_bytes)[list(np.load(b_bytes).files)[0]]
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid .npz files")
	if A.shape != B.shape:
		raise HTTPException(status_code=400, detail="Matrix shapes do not match")
	try:
		_, elapsed = gpu_matrix_add(A, B)
	except cuda.CudaSupportError:
		raise HTTPException(status_code=500, detail="CUDA device not found or not available")
	return {
		"matrix_shape": list(A.shape),
		"elapsed_time": round(elapsed, 6),
		"device": "GPU"
	}

@app.get("/health")
def health():
	return {"status": "ok"}

@app.get("/gpu-info")
def gpu_info():
	try:
		result = subprocess.run([
			"nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"
		], capture_output=True, text=True, check=True)
		gpus = []
		for line in result.stdout.strip().split("\n"):
			idx, used, total = line.split(",")
			gpus.append({
				"gpu": idx.strip(),
				"memory_used_MB": int(used.strip()),
				"memory_total_MB": int(total.strip())
			})
		return {"gpus": gpus}
	except Exception:
		raise HTTPException(status_code=500, detail="nvidia-smi not available or failed")
