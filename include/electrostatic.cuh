#pragma once
#include "matrixfree.cuh"
#include "model.cuh"

class ElectroStatic: public Matrixfree {
public:
	ElectroStatic(BeadsModel& model);
	~ElectroStatic() {
		if (d_coefC_pre) cudaFree(d_coefC_pre);
	};

	double epp;
	double epw;
	std::vector<double> EFinf;
	void compute_matrix(cuDoubleComplex* Mc, int icol, int jcol, int irow, int jrow, bool c_major) const override;
	void get_matrix(std::vector<std::complex<double>>& Mcc, bool c_major) const override;
	void right_side_F(cuDoubleComplex* y, int i, int j) const override;
	void right_side_F_mkl(MKL_Complex16* y, int i, int j) const override;
	void matmult(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta) const override;

	void boundary_condition();
private:
	double* d_coefC_pre = nullptr;
	std::vector<std::complex<double>> Be;
};