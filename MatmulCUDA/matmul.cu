#include "matmul.cuh"
#include "cuda_call.h"
#include "constant.h"
#include <assert.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
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
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int blockRow = blockIdx.y;  // index within a grid
	int blockCol = blockIdx.x;  // index within a grid
	int tileRow = threadIdx.y;  // index within a tile
	int tileCol = threadIdx.x;  // index within a tile
	int A_tile_base, B_tile_base;
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Bsub[TILE_SIZE][TILE_SIZE];


	float accu = 0;
	if (row < n && col < n) {
		for (int blk = 0; blk < n / TILE_SIZE; blk++) {
			A_tile_base = TENSOR_SIZE * BLOCK_SIZE * blockRow + BLOCK_SIZE * blk;  // A: fix row, change col
			B_tile_base = TENSOR_SIZE * BLOCK_SIZE * blk + BLOCK_SIZE * blockCol;  // B: fix col, change row

			// for each tile, copy memory first
			Asub[tileRow][tileCol] = a[A_tile_base + IDX2R(row, col, TENSOR_SIZE)];
			Bsub[tileRow][tileCol] = b[B_tile_base + IDX2R(row, col, TENSOR_SIZE)];
			__syncthreads();
			
		}
		for (int i = 0; i < BLOCK_SIZE; i++) {
			accu += Asub[tileRow][i] * Bsub[i][tileCol];
		}
		__syncthreads();
	}
	c[TENSOR_SIZE * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol + TENSOR_SIZE * row + col] = accu;
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

