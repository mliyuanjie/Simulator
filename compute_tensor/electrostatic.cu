#include "electrostatic.cuh"
#include <boost/math/special_functions/legendre.hpp>
__constant__ int lm2idx_gpu[216];
__inline__ __device__ double warpReduceSum(double val, int width) {
#pragma unroll
	for (int offset = width >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset, width);
	}
	return val;
}
__global__ void _matmult_cuda_F_e(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta, int Np, double* d_coefC_pre, double3* xyz, double* rp, double epp, double epw) {
	__shared__ cuDoubleComplex tmp[2];
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int iconta;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x;
	double _pre;
	cuDoubleComplex res = {0, 0};
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		iconta = blockIdx.y + i * n_harm;
		dimj = n_harm * gridDim.x;
		int idx = (blockIdx.y + i * n_harm) * dimj;
		double coeff = (epp * lp + epw * (lp + 1)) / (epw - epp) / lp / pow(rp[i], 2 * lp + 1);
		res = cuCmul(make_cuDoubleComplex(coeff, 0.0), x[iconta]);
		for (int tmpi = 0; tmpi < 2; tmpi++) {
			tmp[tmpi] = make_cuDoubleComplex(0, 0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= n_harm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
	}
	for (int j = 0; j < Np; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = tid + j * n_harm;
			double coeff = -_pre;
			coeff *= legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff *= pow(xyz_polar.x, -l - lp - 1);
			cuDoubleComplex coeffz = coeff * cuexp(0, (m - mp) * xyz_polar.z);
			res = cuCadd(res, cuCmul(coeffz, x[jconta]));
			
		}
	}

	int size_warp = (tid >= 32) ? n_harm - 31 : 32;
	res.x = warpReduceSum(res.x, size_warp);
	res.y = warpReduceSum(res.y, size_warp);
	if (tid % warpSize == 0) {
		tmp[tid / warpSize] = res;
	}
	__syncthreads();
	if (tid == 0) {
		y[iconta] = cuCadd(beta * y[iconta], alpha * cuCadd(tmp[0], tmp[1]));
	}
}

__global__ void _matmult_cuda_F_i(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta, int Np, double* d_coefC_pre, double3* xyz, double* rp) {
	__shared__ cuDoubleComplex tmp[2];
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int iconta;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x;
	double _pre;
	cuDoubleComplex res = { 0, 0 };
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		iconta = blockIdx.y + i * n_harm;
		dimj = n_harm * gridDim.x;
		int idx = (blockIdx.y + i * n_harm) * dimj;
		double coeff = 1.0 / pow(rp[i], 2 * lp + 1);
		res = cuCmul(make_cuDoubleComplex(coeff, 0.0), x[iconta]);
		for (int tmpi = 0; tmpi < 2; tmpi++) {
			tmp[tmpi] = make_cuDoubleComplex(0, 0);
		}
	}
	__syncthreads();
	l = lm2idx_gpu[2 * tid];
	m = lm2idx_gpu[2 * tid + 1];
	int pre_id = mp + lp + (lp - 1) * (lp + 1);
	pre_id *= n_harm;
	pre_id += m + l + (l - 1) * (l + 1);
	_pre = d_coefC_pre[pre_id];
	for (int j = 0; j < Np; j++) {
		if (i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = tid + j * n_harm;
			double coeff = _pre;
			coeff *= legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff *= pow(xyz_polar.x, -l - lp - 1);
			cuDoubleComplex coeffz = coeff * cuexp(0, (m - mp) * xyz_polar.z);
			res = cuCadd(res, cuCmul(coeffz, x[jconta]));

		}
	}

	int size_warp = (tid >= 32) ? n_harm - 31 : 32;
	res.x = warpReduceSum(res.x, size_warp);
	res.y = warpReduceSum(res.y, size_warp);
	if (tid % warpSize == 0) {
		tmp[tid / warpSize] = res;
	}
	__syncthreads();
	if (tid == 0) {
		y[iconta] = cuCadd(beta * y[iconta], alpha * cuCadd(tmp[0], tmp[1]));
	}
}

__global__ void _compute_matrix_cuda_F_e(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double* d_coefC_pre, double3* xyz, double* rp, double epp, double epw) {
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int iconta;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double _pre;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		iconta = blockIdx.y + (i - si) * n_harm;
		dimj = n_harm * gridDim.x;
		if (sj <= i && ej > i) {
			int idx = (blockIdx.y + (i - sj) * n_harm) * dimj;
			double coeff = (epp * lp + epw * (lp + 1)) / (epw - epp) / lp / pow(rp[i], 2 * lp + 1);
			Mc[idx + iconta] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= n_harm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int idx = (tid + (j - sj) * n_harm) * dimj;
			double coeff = -_pre;
			coeff *= legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff *= pow(xyz_polar.x, -l - lp - 1);
			cuDoubleComplex coeffz = coeff * cuexp(0, (m - mp) * xyz_polar.z);
			Mc[idx + iconta] = coeffz;
		}
	}
}

__global__ void _compute_matrix_cuda_C_e(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double* d_coefC_pre, double3* xyz, double* rp, double epp, double epw) {
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int idx;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double _pre;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		dimj = n_harm * (ej - sj);
		idx = (blockIdx.y + (i - si) * n_harm) * dimj;
		if (sj <= i && ej > i) {
			int iconta = blockIdx.y + (i - sj) * n_harm;
			double coeff = (epp * lp + epw * (lp + 1)) / (epw - epp) / lp / pow(rp[i], 2 * lp + 1);
			Mc[idx + iconta] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= n_harm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
		
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = tid + (j - sj) * n_harm;
			double coeff_real = -_pre;
			coeff_real *= legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff_real *= pow(xyz_polar.x, -l - lp - 1);
			cuDoubleComplex coeff = coeff_real * cuexp(0, (m - mp) * xyz_polar.z);
			Mc[idx + jconta] = coeff;
		}
	}
}

__global__ void _compute_matrix_cuda_F_i(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double* d_coefC_pre, double3* xyz, double* rp) {
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int iconta;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double _pre;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		iconta = blockIdx.y + (i - si) * n_harm;
		dimj = n_harm * gridDim.x;
		if (sj <= i && ej > i) {
			int idx = (blockIdx.y + (i - sj) * n_harm) * dimj;
			double coeff = 1.0 / pow(rp[i], 2 * lp + 1);
			Mc[idx + iconta] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= n_harm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int idx = (tid + (j - sj) * n_harm) * dimj;
			double coeff_real = _pre;
			coeff_real = coeff_real * legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff_real = coeff_real * pow(xyz_polar.x, -l - lp - 1.0);
			cuDoubleComplex coeff = cuexp(0, (m - mp) * xyz_polar.z * coeff_real);
			Mc[idx + iconta] = coeff;
		}
	}
}

__global__ void _compute_matrix_cuda_C_i(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double* d_coefC_pre, double3* xyz, double* rp) {
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int idx;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double _pre;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		dimj = n_harm * (ej - sj);
		idx = (blockIdx.y + (i - si) * n_harm) * dimj;
		if (sj <= i && ej > i) {
			int iconta = blockIdx.y + (i - sj) * n_harm;
			double coeff = 1.0 / pow(rp[i], 2 * lp + 1);
			Mc[idx + iconta] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= n_harm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = tid + (j - sj) * n_harm;
			double coeff_real = _pre;
			coeff_real *= legendre(l + lp, abs(m - mp), xyz_polar.y);
			coeff_real *= pow(xyz_polar.x, -l - lp - 1.0);
			cuDoubleComplex coeff = cuexp(0, (m - mp) * xyz_polar.z * coeff_real);
			Mc[idx + jconta] = coeff;
		}
	}
}

ElectroStatic::ElectroStatic(BeadsModel& model_in) : Matrixfree(model_in) {
	cols = 3;
	nharm = Nc * (Nc + 2);
	epp = model.epp;
	epw = model.epw;
	dim = Nc * (Nc + 2) * Np;

	//build factori function in array and calculate the pre coeffC;
	std::vector<double> _facto = std::vector<double>(4 * Nc + 1, 0);
	std::vector<std::vector<double>> _coefC_pre = std::vector<std::vector<double>>(nharm, std::vector<double>(nharm, 0));
	_facto[0] = 1.0;
	for (int i = 1; i < _facto.size(); i++) {
		_facto[i] = std::tgamma(i + 1);
	}
	for (int lp = 1; lp <= Nc; lp++) {
		for (int mp = -lp; mp <= lp; mp++) {
			int i = mp + lp + (lp - 1) * (lp + 1);
			for (int l = 1; l <= Nc; l++) {
				for (int m = -l; m <= l; m++) {
					int j = m + l + (l - 1) * (l + 1);
					double un = 1.0;
					if (m - mp < 0) {
						un = pow(-1.0, m - mp) * _facto[lp + l - abs(mp - m)] / _facto[lp + l + abs(mp - m)];
					}
					_coefC_pre[i][j] = pow(-1, mp + lp) * _facto[lp + l - m + mp] / (_facto[l - m] * _facto[mp + lp]) * un;
				}
			}
		}
	}
	cudaMalloc(&d_coefC_pre, sizeof(double) * nharm * nharm);
	for(int i = 0; i <nharm; i++)
		cudaMemcpy((void*)(d_coefC_pre + i * nharm), (void*)_coefC_pre[i].data(), sizeof(double) * nharm, cudaMemcpyHostToDevice);

	EFinf = std::vector<double>(9, 0);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			EFinf[i * 3 + j] = kdelta(i + 1, j + 1);
		}
	}
	std::vector<int> lm2idx(2 * Nc * (Nc + 2), 0);
	for (int l = 1; l <= Nc; l++) {
		for (int m = -l; m <= l; m++) {
			int idx = m + l + l * l - 1;
			lm2idx[idx * 2] = l;
			lm2idx[idx * 2 + 1] = m;
		}
	}
	cudaMemcpyToSymbol(lm2idx_gpu, lm2idx.data(), lm2idx.size() * sizeof(int), 0, cudaMemcpyHostToDevice);
	boundary_condition();
}

void ElectroStatic::boundary_condition() {
	std::complex<double> xj = { 0.0, 1.0 };
	Be = std::vector<std::complex<double>>(dim * 3, 0);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < Np; i++) {
			for (int l = 1; l < Nc; l++) {
				for (int m = -l; m <= l; m++) {
					int iconta = i * nharm + l * l - 1 + l + m;
					if (m == 0 && l == 1) {
						Be[iconta + j * dim] = -EFinf[j * 3 + 2];
					}
					else if (m == 1 && l == 1) {
						Be[iconta + j * dim] = (EFinf[j * 3] - xj * EFinf[j * 3 + 1]) / 2.0;
					}
					else if (m == -1 && l == 1) {
						Be[iconta + j * dim] = -EFinf[j * 3] - xj * EFinf[j * 3 + 1];
					}
				}
			}
		}
	}
}

void ElectroStatic::right_side_F(cuDoubleComplex* y, int icol, int jcol) const {
	if (Be.size() != cols * dim)
		throw std::runtime_error("Error: right hand vector not initialize");
	int dimj = nharm * (jcol - icol);
	int offset = icol * nharm;
	cudaMemcpy2D((void*)y, dimj * sizeof(cuDoubleComplex),
		(void*)(Be.data() + offset), dim * sizeof(cuDoubleComplex),
		dimj * sizeof(cuDoubleComplex), cols, cudaMemcpyHostToDevice);
	return;
}
void ElectroStatic::right_side_F_mkl(MKL_Complex16* y, int icol, int jcol) const {
	if (Be.size() != cols * dim)
		throw std::runtime_error("Error: right hand vector not initialize");
	int dimj = nharm * (jcol - icol);
	int offset = icol * nharm;
	mkl_zomatcopy('C', 'N', dimj, cols, { 1.0, 0.0 },
		reinterpret_cast<const MKL_Complex16*>(Be.data() + offset), dim,
		y, dimj);
	return;
}
void ElectroStatic::compute_matrix(cuDoubleComplex* Mc, int icol, int jcol, int irow, int jrow, bool c_major) const {
	dim3 blockDims(nharm, 1);
	dim3 gridDims(jcol - icol, nharm);
	int dimx = (jcol - icol) * nharm;
	int dimy = (jrow - irow) * nharm;
	cudaMemset(Mc, 0, dimx * dimy * sizeof(cuDoubleComplex));
	if (c_major)
		_compute_matrix_cuda_C_e << <gridDims, blockDims >> > (Mc, icol, irow, jrow, Np, d_coefC_pre, model.d_xyz, model.d_rp, epp, epw);
	else
		_compute_matrix_cuda_F_e << <gridDims, blockDims >> > (Mc, icol, irow, jrow, Np, d_coefC_pre, model.d_xyz, model.d_rp, epp, epw);
}

void ElectroStatic::get_matrix(std::vector<std::complex<double>>& Mcc, bool c_major) const {
	cuDoubleComplex* Mc;
	cudaMalloc(&Mc, dim * dim * sizeof(cuDoubleComplex));
	compute_matrix(Mc, 0, Np, 0, Np, c_major);
	cudaMemcpy(Mcc.data(), Mc, dim * dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

void ElectroStatic::matmult(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta) const {
	//int num_thread = 32;
	//int harm_num = Nc * (Nc + 2);
	//if (harm_num > 32)
	//	num_thread = 64;
	dim3 blockDims(nharm, 1);
	dim3 gridDims(Np, nharm);
	for (int i = 0; i < cols; i++) {
		_matmult_cuda_F_i <<<gridDims, blockDims >>> (x + i * dim, alpha, y + i * dim, beta, Np, d_coefC_pre, model.d_xyz, model.d_rp);
	}
}