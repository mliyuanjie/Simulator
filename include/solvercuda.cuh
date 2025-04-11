#pragma once
#include "matrixfree.cuh"
#include <cusolverDn.h>
#include "mkl.h"

class Solver {
public:
	Solver(Matrixfree& _matrix, int _solver_type) : matrix(_matrix) {
		//0 direct solver on gpu
		//1 direct solver on CPU
		//2 schur + lu on gpu 
		//3 mix on gpu and cpu
		cols = matrix.cols;
		Np = matrix.Np;
		Nc = matrix.Nc;
		nharm = matrix.nharm;
		dim = matrix.dim;
		solver_type = _solver_type;
		if (solver_type == 2) {
			NpS = Np / 2;
			dimS = NpS * nharm;
			NpA = Np - NpS;
			dimA = NpA * nharm;
			//allocate the matmult space
			sizeS = size_t(dimA * dimS) * sizeof(cuDoubleComplex);
			sizeA = size_t(dimA * dimA) * sizeof(cuDoubleComplex);
			sizeV = size_t(cols * dim) * sizeof(cuDoubleComplex);
			offsetS = dimA * cols;
			cublasCreate(&handle);
			cublasSetStream(handle, stream);
			cudaMalloc(&d_A, sizeA);
			cudaMalloc(&d_B, sizeS);
			cudaMalloc(&d_y, sizeV);
			cudaMalloc(&d_x, sizeV);
			//LU work space
			cusolverDnCreate(&cusolverH);
			cusolverDnZgetrf_bufferSize(cusolverH, dimA, dimA, d_A, dimA, &lu_work_size);
			if (lu_work_size < dimS * dimA) {
				lu_work_size = dimS * dimA;
			}
			cudaMalloc(&lu_work, lu_work_size * sizeof(cuDoubleComplex));
			cudaMalloc(&devIpiv, dimA * sizeof(int));
			cudaMalloc(&devInfo, sizeof(int));
		}
		else if (solver_type == 0) {
			//allocate the matmult space
			sizeA = size_t(dim * dim) * sizeof(cuDoubleComplex);
			sizeV = size_t(cols * dim) * sizeof(cuDoubleComplex);
			cublasCreate(&handle);
			cublasSetStream(handle, stream);
			cudaMalloc(&d_A, sizeA);
			cudaMalloc(&d_y, sizeV);
			cudaMalloc(&d_x, sizeV);
			//LU work space
			cusolverDnCreate(&cusolverH);
			cusolverDnZgetrf_bufferSize(cusolverH, dim, dim, d_A, dim, &lu_work_size);
			cudaMalloc(&lu_work, lu_work_size * sizeof(cuDoubleComplex));
			cudaMalloc(&devIpiv, dim * sizeof(int));
			cudaMalloc(&devInfo, sizeof(int));
		}
		else if (solver_type == 1) {
			//maxbeads 250
			sizeA = size_t(100 * 100 * nharm * nharm) * sizeof(cuDoubleComplex);
			sizeV = size_t(cols * dim) * sizeof(cuDoubleComplex);
			cudaMalloc(&d_A, sizeA);
			cudaMalloc(&d_y, sizeV);
			cudaMalloc(&d_x, sizeV);
			c_A = new MKL_Complex16[dim * dim];
			c_x = new MKL_Complex16[dim * cols];
		}
	};
	~Solver() {
		if (d_A) cudaFree(d_A);
		if (d_B) cudaFree(d_B);
		if (d_x) cudaFree(d_x);
		if (d_y) cudaFree(d_y);
		if (handle) cublasDestroy(handle);
		if (stream) cudaStreamDestroy(stream);
		if (lu_work) cudaFree(lu_work);
		if (devIpiv) cudaFree(devIpiv);
		if (devInfo) cudaFree(devInfo);
		if (cusolverH) cusolverDnDestroy(cusolverH);
		delete[] c_A;
		delete[] c_x;

	};

	void set_x(std::vector<std::complex<double>>& x_ptr) {
		cudaMemcpy(d_x, x_ptr.data(), sizeV, cudaMemcpyHostToDevice);
	};
	void get_y(std::vector<std::complex<double>>& y_ptr) {
		if (y_ptr.size() != matrix.cols * matrix.dim)
			y_ptr.resize(matrix.cols * matrix.dim);
		cudaMemcpy(y_ptr.data(), d_y, sizeV, cudaMemcpyDeviceToHost);
	};
	void get_x(std::vector<std::complex<double>>& x_ptr) {
		if (x_ptr.size() != matrix.cols * matrix.dim)
			x_ptr.resize(matrix.cols * matrix.dim);
		cudaMemcpy(x_ptr.data(), d_x, sizeV, cudaMemcpyDeviceToHost);
	};
	void set_y(std::vector<std::complex<double>>& y_ptr) {
		cudaMemcpy(d_y, y_ptr.data(), sizeV, cudaMemcpyHostToDevice);
	};
	void solve(const char* str);
	void matmult(double alpha, double beta);
	void verify(std::string& fn);
	void clear();
	void initial();
	Matrixfree& matrix;
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	cuDoubleComplex* d_A = nullptr;
	cuDoubleComplex* d_B = nullptr;
	cuDoubleComplex* d_x = nullptr;
	cuDoubleComplex* d_y = nullptr;
	size_t dim = 0;
	int dimS = 0;
	int dimA = 0;
	size_t sizeS = 0;
	size_t sizeA = 0;
	size_t sizeV = 0;
	int nharm = 0;
	int Np = 0;
	int NpS = 0;
	int NpA = 0;
	int cols = 0;
    int Nc = 0;
	int level = 0;
	int offsetS = 0;
	int solver_type = 0;

	MKL_Complex16* c_A = nullptr;
	MKL_Complex16* c_x = nullptr;

	//LU information and data 
	int* devIpiv = nullptr;
	int* devInfo = nullptr;
	int lu_work_size;
	cuDoubleComplex* lu_work = nullptr;
	cusolverDnHandle_t cusolverH;
	private:
		void Schur();
		void LU();
		void LU_mkl();
};