#include <iostream>
#include "matmul.cuh"
#include <vector>
#include "constant.h"
#include <chrono>
#include <cmath>
#include "cuda_call.h"
#include <cublas_v2.h>

#ifdef TEST_CUBLAS
int cublasMatmul(float_tensor &A, float_tensor &B, float_tensor &result, cublasHandle_t cublas_handle)
{
	float alpha = 1; float beta = 0;
	// cublas uses column major, meaning that devA = hostA(T). Such that, devA(T) * devB(T) = devAB(T) = hostAB.
	// this means, doing gemm(B,A) = AB in host memory. Note though, cublasGemmEx uses device memory
	cublasSgemm(cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		result.dims()[0], // m
		result.dims()[1], // n
		A.dims()[1],      // k
		&alpha,
		A.begin(), A.dims()[0],
		B.begin(), B.dims()[0],
		&beta,
		result.begin(), result.dims()[0]);

#ifndef DISABLE_CUDA_DEVICE_SYNC
	cudaDeviceSynchronize();
#endif // DISABLE_CUDA_DEVICE_SYNC
	return 0;
}

double test_cublas(int matrix_size) {
	int data_count = matrix_size * matrix_size;
	double flop = 2 * pow(matrix_size, 3);
	int memsize = sizeof(float) * data_count;
	float *data = new float[data_count];
	std::fill_n(data, data_count, MEMSET_VAL);

	std::vector<int> dim{ matrix_size , matrix_size };

	float_tensor a(dim);
	float_tensor b(dim);
	float_tensor c(dim);

	auto begin = std::chrono::high_resolution_clock::now();
	a.memcpy2device((char *)data);
	b.memcpy2device((char *)data);
	auto end = std::chrono::high_resolution_clock::now();

	// setting up cublas
	cublasHandle_t handle;
	cublasCreate(&handle);

	// commence the calculation
	begin = std::chrono::high_resolution_clock::now();
	cublasMatmul(a, b, c, handle);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();

	// measure flops perf
	auto matmul_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	// (2p-1)*m*n flop
	std::cout << " - CuBLAS Matmul: " << flop / (double)matmul_time_ns << "GFLOPS" << std::endl;
	delete[] data;

	CUDA_CALL(cudaGetLastError());
	return flop / (double)matmul_time_ns;
}

#endif // TEST_CUBLAS

double test_matmul(int matrix_size) {
	int data_count = matrix_size * matrix_size;
	double flop = 2 * pow(matrix_size, 3);
	int memsize = sizeof(float) * data_count;
	float *data = new float[data_count];
	std::fill_n(data, data_count, MEMSET_VAL);

	std::vector<int> dim{ matrix_size , matrix_size };

	float_tensor a(dim);
	float_tensor b(dim);
	float_tensor c(dim);

	auto begin = std::chrono::high_resolution_clock::now();
	a.memcpy2device((char *)data);
	b.memcpy2device((char *)data);
	auto end = std::chrono::high_resolution_clock::now();

	begin = std::chrono::high_resolution_clock::now();
	CUDA_CALL(matmul(a, b, c));
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	auto matmul_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	// (2p-1)*m*n flop
	std::cout << " - Matmul: " << flop / (double)matmul_time_ns << "GFLOPS" << std::endl;
	delete[] data;

	CUDA_CALL(cudaGetLastError());
	return flop / (double)matmul_time_ns;
}

#ifdef MUL
int main() {
	#include <cublas_v2.h>
	int size = TENSOR_SIZE;
	for (int i = 0; i < 4; i++) {
		std::cout << "Test size: " << size << std::endl;
		test_matmul(size);
#ifdef TEST_CUBLAS
		test_cublas(size);
#endif // TEST_CUBLAS
		size <<= 1;
	}
exit:
	std::cin.get();
    return 0;
}
#endif