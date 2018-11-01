#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constant.h"
#include "cuda_call.h"
#include <iostream>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

void print_matrix(const Matrix &m) {
	for (int i = 0; i < m.height; i++) {
		for (int j = 0; j < m.width; j++) {
			std::cout << m.elements[i * m.width + j] << ", ";
		}
		std::cout << std::endl;
	}
}

// Thread block size
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);
	CUDA_CALL(cudaGetLastError());

	// Read C from device memory
	CUDA_CALL(cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost));
	// Free device memory
	CUDA_CALL(cudaFree(d_A.elements));
	CUDA_CALL(cudaFree(d_B.elements));
	CUDA_CALL(cudaFree(d_C.elements));
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]
		* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}

#ifndef MUL
int main() {
	int data_count = TENSOR_SIZE * TENSOR_SIZE;
	float *data = new float[data_count];
	float *result_data = new float[data_count];
	std::fill_n(data, data_count, 10.4);

	Matrix A = { TENSOR_SIZE, TENSOR_SIZE, data };
	Matrix B = { TENSOR_SIZE, TENSOR_SIZE, data };
	Matrix C = { TENSOR_SIZE, TENSOR_SIZE, result_data };
	MatMul(A, B, C);
	delete[] data;
	delete[] result_data;

	std::cin.get();
}
#endif