#include "matmul.cuh"
#include "cuda_call.h"
#include "constant.h"
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


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
	int blockRow = blockIdx.y;  // index within a grid
	int blockCol = blockIdx.x;  // index within a grid
	int tileRow = threadIdx.y;  // index within a tile
	int tileCol = threadIdx.x;  // index within a tile

	int row = blockRow * TILE_SIZE + tileRow;
	int col = blockCol * TILE_SIZE + tileCol;

	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Bsub[TILE_SIZE][TILE_SIZE];


	float accu = 0;
	for (int blk = 0; blk < n / TILE_SIZE; blk++) {
		// for each tile, copy memory first
		Asub[tileRow][tileCol] = a[row * n + (blk * TILE_SIZE + tileCol)];
		Bsub[tileRow][tileCol] = b[(blk * TILE_SIZE + tileRow) * n + col];
		__syncthreads();

		for (int i = 0; i < TILE_SIZE; i++) {
			accu += Asub[tileRow][i] * Bsub[i][tileCol];
		}

		__syncthreads();
	}
	c[row * n + col] = accu;
}

__global__ void matmul_kernel_opt(float *c, const float *a, const float *b, int n) {
	// this implementation depends on the size:
	// OPT_TILE_SIZE		16
	// OPT_WIDTH			64	// keep same as the thread number
	// OPT_BLOCK_X_SIZE		16
	// OPT_BLOCK_Y_SIZE		4


	int by = blockIdx.y;  // index within a grid
	int bx = blockIdx.x;  // index within a grid
	int ty = threadIdx.y;  // index within a tile
	int tx = threadIdx.x;  // index within a tile

	__shared__ float Asub[OPT_TILE_SIZE][OPT_TILE_SIZE];
	float c[OPT_TILE_SIZE] = { 0 };

	// load A into shared memory
	for (int i = 0; i < 4; i++) {
		// A[(i * 4 + ty) + BLOCK_SIZE * tx] = A[a + wA * (i * 4 + ty) + tx];
		Asub[(i * 4 + tx)][ty] = a[a + wA * (i * 4 + ty) + tx];
	}


}


cudaError_t matmul(float_tensor &a, float_tensor &b, float_tensor &c) {
	assert(a.dims()[1] == b.dims()[0]);
	int inner_dim = a.dims()[1];
	 
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(GRID(c.dims()[0], blockDim.x), GRID(c.dims()[1], blockDim.y));
	// std::cout << gridDim.x << ' ' << gridDim.y << std::endl;
	matmul_kernel <<<gridDim, blockDim >>> (c.begin(), a.begin(), b.begin(), inner_dim);

	return cudaGetLastError();
}

