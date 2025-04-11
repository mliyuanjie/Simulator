#pragma once
#include "matrixfree.cuh"
#include "model.cuh"




class Hydrodynamic: public Matrixfree {
public:
	Hydrodynamic(BeadsModel& model);
	~Hydrodynamic() {
		if (d_YS) cudaFree(d_YS);
		if (d_coefC_pre) cudaFree(d_coefC_pre);
		if (d_cnlm) cudaFree(d_cnlm);
	};

	void compute_matrix(cuDoubleComplex* Mc, int icol, int jcol, int irow, int jrow, bool cmajor) const override;
	void get_matrix(std::vector<std::complex<double>>& Mcc, bool c_major) const override;
	void right_side_F(cuDoubleComplex* y, int i, int j) const override;
	void right_side_F_mkl(MKL_Complex16* y, int i, int j) const override;
	void matmult(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta) const override;
	
	void set_slip_condition(std::vector<std::complex<double>>& Be, std::vector<std::complex<double>>& g);
private:
	double9* d_YS = nullptr;
	double* d_coefC_pre = nullptr;
	double* d_cnlm = nullptr;
	std::vector<std::vector<double>> _cnlm;
	int offset_cnlm;
	std::vector<std::vector<double>> _coefC_pre;
	std::vector<std::complex<double>> yb;
};