#pragma once
#include <vector>
#include <complex>
#include <iostream>
#include <math_constants.h> 
#include <math.h> 
#include <cmath>
#include <cuda.h>

#include <device_launch_parameters.h>

template<typename T>
class CudaMatrix {
public:
	size_t rows;
	size_t cols;
	T* data;

    CudaMatrix(size_t rows, size_t cols)
        : rows(rows), cols(cols), d_data(nullptr)
    {
        size_t size = rows * cols * sizeof(T);
        cudaError_t err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~CudaMatrix() {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;

    CudaMatrix& operator=(CudaMatrix&& other) noexcept {
        if (this != &other) {
            if (d_data) {
                cudaFree(d_data);
            }
            rows = other.rows;
            cols = other.cols;
            d_data = other.d_data;
            other.d_data = nullptr;
            other.rows = other.cols = 0;
        }
        return *this;
    }

    size_t sizeInBytes() const {
        return rows * cols * sizeof(T);
    }

    void copy(std::vector<T>& ptr_c) const {
        if (host_vector.size() != rows * cols) {
            host_vector.resize(rows * cols);
        }
        cudaError_t err = cudaMemcpy(host_vector.data(), d_data, sizeInBytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        }
    }
};
