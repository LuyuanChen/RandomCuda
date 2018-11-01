#include "matmul.cuh"
#include "cuda_call.h"
#include "constant.h"
#include <assert.h>

#define IDX2R(i,j,ld) (((i)*(ld))+(j))  // row major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // column major
#define GRID(n, blk_sz) ((n % blk_sz) == 0 ? n / blk_sz : n / blk_sz + 1)

/// <summary>
/// Kernel for calculating the matrix multiplication
/// </summary>
/// <param name="c">Result, mxn</param>
/// <param name="a">Input, mxk</param>
/// <param name="b">Input, kxn</param>
/// <param name="n">dim</param>
/// <returns>void</returns>
__global__ void matmul_kernel(float *c, const float *a, const float *b, int n) {
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	float accu = 0;
	if (row < n && col < n) {
		for (int i = 0; i < n; i++) {
			accu += a[IDX2R(row, i, n)] * b[IDX2R(i, col, n)];
		}
	}
	c[IDX2R(row, col, n)] = accu;
}

cudaError_t matmul(float_tensor &a, float_tensor &b, float_tensor &c) {
	assert(a.dims()[1] == b.dims()[0]);
	int inner_dim = a.dims()[1];

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(GRID(c.dims()[0], blockDim.x), GRID(c.dims()[1], blockDim.y));
	std::cout << gridDim.x << ' ' << gridDim.y << std::endl;
	matmul_kernel <<<gridDim, blockDim >>> (c.begin(), a.begin(), b.begin(), inner_dim);
	return cudaGetLastError();
}

