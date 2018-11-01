#include <iostream>
#include "matmul.cuh"
#include <vector>
#include "constant.h"
#include <chrono>
#include <cmath>
#include "cuda_call.h"

#ifdef MUL
int main() {
	int data_count = TENSOR_SIZE * TENSOR_SIZE;
	double flop = 2 * pow(TENSOR_SIZE, 3);
	int memsize = sizeof(float) * data_count;
	float *data = new float[data_count];
	std::fill_n(data, data_count, 10.4);

	std::vector<int> dim{ TENSOR_SIZE , TENSOR_SIZE };
	
	float_tensor a(dim);
	float_tensor b(dim);
	float_tensor c(dim);

	auto begin = std::chrono::high_resolution_clock::now();
	a.memcpy2device((char *)data);
	b.memcpy2device((char *)data);

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Memcpy: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;

	begin = std::chrono::high_resolution_clock::now();
	CUDA_CALL(matmul(a, b, c));
	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();
	auto matmul_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	// (2p-1)*m*n flop
	std::cout << "Matmul: " << matmul_time_ns << "ns " << flop << " flop" << std::endl;
	std::cout << "Matmul: " << flop / (double)matmul_time_ns << "GFLOPS" << std::endl;
	// std::cout << c << std::endl;
	delete[] data;

	CUDA_CALL(cudaGetLastError());
exit:
	std::cin.get();
    return 0;
}
#endif