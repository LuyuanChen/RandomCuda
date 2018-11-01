#pragma once
#ifndef TENSOR_H_
#define TENSOR_H

#include <assert.h>
#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <curand.h>
template <typename T>
class Tensor {
public:
	std::vector<int> dims_;
	int size_;

	struct deleteCudaPtr {
		void operator()(T *p) const {
			cudaFree(p);
		}
	};

	std::shared_ptr<T> ptr_;

	Tensor() {}

	Tensor(std::vector<int> dims) : dims_(dims) {
		T* tmp_ptr;
		size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
		cudaMalloc(&tmp_ptr, sizeof(T) * size_);
		ptr_.reset(tmp_ptr, deleteCudaPtr());
	}

	Tensor(std::vector<int> dims, char *data) : Tensor(dims) {
		cudaMemcpy(this->begin(), data, sizeof(T) * size_, cudaMemcpyHostToDevice);
	}

	T* begin() const { return ptr_.get(); }
	T* end()   const { return ptr_.get() + size_; }
	int size() const { return size_; }
	int byte_size() const { return size_ * sizeof(T); }
	std::vector<int> dims() const { return dims_; }
	void fill_float(float num) {
		thrust::fill(thrust::device_ptr<T>(this->begin()),
			thrust::device_ptr<T>(this->end()), num);
	}
	void randomize(curandGenerator_t curand_gen, float mean, float stddev) {
		curandGenerateNormal(curand_gen, this->begin(), this->size(), mean, stddev);
	}

	void memcpy2device(char *src, int stream = -1) {
		if (stream > 0) {
			cudaMemcpyAsync(this->begin(), src, this->byte_size(), cudaMemcpyHostToDevice, (cudaStream_t)stream);
		} else {
			cudaMemcpy(this->begin(), src, this->byte_size(), cudaMemcpyHostToDevice);
		}
	}
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
	T* tmp = new T[t.size_];
	cudaMemcpy(tmp, t.ptr_.get(), t.size_ * sizeof(T), cudaMemcpyDeviceToHost);
	std::string toPrint("");
	for (size_t i = 0; i < t.dims_[0]; i++) {
		for (size_t j = 0; j < t.dims_[1]; j++) {

			toPrint += std::to_string(tmp[i * t.dims_[1] + j]);
			toPrint += ",";
		}
		toPrint += '\n';
	}
	delete[] tmp;
	return os << toPrint << std::endl;
}


typedef Tensor<float> float_tensor;

#endif