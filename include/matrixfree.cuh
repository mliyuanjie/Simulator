#pragma once
#include <vector>
#include <complex>
#include <iostream>
#include <math_constants.h> 
#include <math.h> 
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "model.cuh"
#include "mkl.h"


struct __host__ __device__ double9 {
	double ii; double ij; double ik;
	double ji; double jj; double jk;
	double ki; double kj; double kk;
};

template <typename T>
inline int kdelta(T i, T j) {
	return (i == j) ? 1 : 0;
}

template <typename T>
inline std::vector<T> crossp(const T& a, const T& b, const T& c, const std::vector<T>& p) {
	std::vector<T> res = { b * p[2] - c * p[1], c * p[0] - a * p[2], a * p[1] - b * p[0] };
	return res;
}

__device__ inline cuDoubleComplex cuexp(double a, double b) {
	double exp_a = exp(a);
	return make_cuDoubleComplex(exp_a * cos(b), exp_a * sin(b));
}

__device__ inline cuDoubleComplex operator*(const double& scalar, const cuDoubleComplex& complex) {
	return make_cuDoubleComplex(scalar * complex.x, scalar * complex.y);
}

__device__ double inline legendre(int l, int m, double x) {
	if (m > l) return 0.0;
	if (l == 0) return 1.0;
	double Pmm = 1.0;
	double somx2 = sqrt((1.0 - x) * (1.0 + x));
	double fact = 1.0;
	for (int i = 1; i <= m; i++) {
		Pmm *= -fact * somx2;
		fact += 2.0;
	}
	if (l == m) return Pmm;
	double Pmmp1 = x * (2.0 * m + 1.0) * Pmm;
	if (l == m + 1) return Pmmp1;
	double Plm = 0.0;
	for (int ll = m + 2; ll <= l; ll++) {
		Plm = ((2.0 * ll - 1.0) * x * Pmmp1 - (ll + m - 1.0) * Pmm) / (ll - m);
		Pmm = Pmmp1;
		Pmmp1 = Plm;
	}
	return Plm;
}

class Matrixfree {
public: 
	int nharm;
	int Np;
	int Nc;
	size_t dim;
	int cols;
	BeadsModel& model;
	Matrixfree(BeadsModel& _model)
		: model(_model), Np(_model.Np), Nc(_model.Nc) {
		nharm = 0;
		dim = 0;
		cols = 0;
	}
	// icol... are the particle number on i or j direction 
	virtual void compute_matrix(cuDoubleComplex* Mc, int icol, int jcol, int irow, int jrow, bool c_major) const = 0;
	virtual void right_side_F(cuDoubleComplex* y, int i, int j) const = 0;
	virtual void right_side_F_mkl(MKL_Complex16* y, int i, int j) const = 0;
	virtual void matmult(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta) const = 0;
	virtual void get_matrix(std::vector<std::complex<double>>& Mcc, bool c_major) const = 0;
};