#include "hydrodynamic_cu.cuh"
#include <boost/math/special_functions/legendre.hpp>
#include "mkl.h"

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)
__constant__ int lm2idx_gpu[216];
__inline__ __device__ double warpReduceSum(double val, int width) {
#pragma unroll
	for (int offset = width >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset, width);
	}
	return val;
}

__global__ void _compute_matrix_cuda_F(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double9* d_YS, double3* xyz, double* rp) {
	__shared__ double3 pow_rp;
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int iconta;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double9 YS;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		//printf("l: %d, m: %d", lm2idx_gpu[37], mp);
		pow_rp = make_double3(pow(rp[i], lp - 1), pow(rp[i], lp), pow(rp[i], lp + 1));
		iconta = 3 * (blockIdx.y + (i - si) * n_harm);
		dimj = 3 * n_harm * gridDim.x;
		if (sj <= i && ej > i) {
			int idx = 3 * (blockIdx.y + (i - sj) * n_harm) * dimj;
			double coeff = lp / pow(rp[i], lp + 2);
			Mc[idx + 1 + iconta] = make_cuDoubleComplex(coeff, 0.0);
			coeff = 1.0 / pow_rp.z;
			Mc[idx + 2 + iconta + dimj] = make_cuDoubleComplex(coeff, 0.0);
			coeff = lp * (lp + 1.0) / pow_rp.y;
			Mc[idx + iconta + 2 * dimj] = make_cuDoubleComplex(coeff, 0.0);
			coeff = -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y;
			Mc[idx + 1 + iconta + 2 * dimj] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int ys_id = mp + lp + (lp - 1) * (lp + 1);
		ys_id *= n_harm;
		ys_id += m + l + (l - 1) * (l + 1);
		YS = d_YS[ys_id];
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int idx = 3 * ((j - sj) * n_harm + tid) * dimj;
			double plm0 = legendre(l + lp - 2, abs(m - mp), xyz_polar.y);
			double plm1 = legendre(l + lp - 1, abs(m - mp), xyz_polar.y);
			double plm2 = legendre(l + lp, abs(m - mp), xyz_polar.y);

			double rij0 = pow(xyz_polar.x, -l - lp - 1.0);
			double rij1 = pow(xyz_polar.x, -l - lp);
			double rij2 = pow(xyz_polar.x, -l - lp + 1.0);

			cuDoubleComplex exp_phi = cuexp(0, (m - mp) * xyz_polar.z);
			cuDoubleComplex exp_phi2 = make_cuDoubleComplex(-exp_phi.y, exp_phi.x);

			double coeff = YS.ki * pow_rp.x * plm2 * rij0;
			Mc[idx + iconta] = coeff * exp_phi;

			coeff = YS.ji * pow_rp.x * plm1 * rij1;
			Mc[idx + iconta + dimj] = coeff * exp_phi2;

			coeff = YS.jj * pow_rp.y * plm2 * rij0;
			Mc[idx + 2 + iconta + dimj] = coeff * exp_phi;

			coeff = (YS.ii * pow_rp.x * plm2 + YS.kk * pow_rp.x * plm0) * rij2 + YS.ik * pow_rp.z * rij0 * plm2;
			Mc[idx + iconta + 2 * dimj] = coeff * exp_phi;

			coeff = YS.jk * pow_rp.z * rij0 * plm2;
			Mc[idx + 1 + iconta + 2 * dimj] = coeff * exp_phi;

			coeff = YS.ij * pow_rp.y * plm1 * rij1;
			Mc[idx + 2 + iconta + 2 * dimj] = coeff * exp_phi2;
		}
	}
}
__global__ void _compute_matrix_cuda_C(cuDoubleComplex* Mc, int si, int sj, int ej, int Np, double9* d_YS, double3* xyz, double* rp) {
	__shared__ double3 pow_rp;
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int idx;
	__shared__ int dimj;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x + si;
	double9 YS;
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		pow_rp = make_double3(pow(rp[i], lp - 1), pow(rp[i], lp), pow(rp[i], lp + 1));
		dimj = 3 * n_harm * (ej - sj);
		idx = 3 * (blockIdx.y + (i - si) * n_harm) * dimj;
		if (sj <= i && ej > i) {
			int iconta = 3 * (blockIdx.y + (i - sj) * n_harm);
			double coeff = lp * (lp + 1.0) / pow_rp.y;
			Mc[idx + iconta + 2] = make_cuDoubleComplex(coeff, 0.0);
			coeff = lp / pow(rp[i], lp + 2);
			Mc[idx + dimj + iconta] = make_cuDoubleComplex(coeff, 0.0);
			coeff = -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y;
			Mc[idx + dimj + iconta + 2] = make_cuDoubleComplex(coeff, 0.0);
			coeff = 1.0 / pow_rp.z;
			Mc[idx + 2 * dimj + iconta + 1] = make_cuDoubleComplex(coeff, 0.0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int ys_id = mp + lp + (lp - 1) * (lp + 1);
		ys_id *= n_harm;
		ys_id += m + l + (l - 1) * (l + 1);
		YS = d_YS[ys_id];
	}
	for (int j = sj; j < ej; j++) {
		if (tid < n_harm && i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = 3 * (tid + (j - sj) * n_harm);
			double plm0 = legendre(l + lp - 2, abs(m - mp), xyz_polar.y);
			double plm1 = legendre(l + lp - 1, abs(m - mp), xyz_polar.y);
			double plm2 = legendre(l + lp, abs(m - mp), xyz_polar.y);

			double rij0 = pow(xyz_polar.x, -l - lp - 1.0);
			double rij1 = pow(xyz_polar.x, -l - lp);
			double rij2 = pow(xyz_polar.x, -l - lp + 1.0);

			cuDoubleComplex exp_phi = cuexp(0, (m - mp) * xyz_polar.z);
			cuDoubleComplex exp_phi2 = make_cuDoubleComplex(-exp_phi.y, exp_phi.x);

			double coeff = YS.ki * pow_rp.x * plm2 * rij0;
			Mc[idx + jconta] = coeff * exp_phi;

			coeff = YS.ji * pow_rp.x * plm1 * rij1;
			Mc[idx + jconta + 1] = coeff * exp_phi2;

			coeff = (YS.ii * pow_rp.x * plm2 + YS.kk * pow_rp.x * plm0) * rij2 + YS.ik * pow_rp.z * rij0 * plm2;
			Mc[idx + jconta + 2] = coeff * exp_phi;

			coeff = YS.jk * pow_rp.z * rij0 * plm2;
			Mc[idx + dimj + jconta + 2] = coeff * exp_phi;

			coeff = YS.jj * pow_rp.y * plm2 * rij0;
			Mc[idx + 2 * dimj + jconta + 1] = coeff * exp_phi;

			coeff = YS.ij * pow_rp.y * plm1 * rij1;
			Mc[idx + 2 * dimj + jconta + 2] = coeff * exp_phi2;
		}
	}
}
__global__ void _matmult_cuda_F(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta, int Np, double9* d_YS, double3* xyz, double* rp) {
	__shared__ cuDoubleComplex tmp[6];
	__shared__ double3 pow_rp;
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int n_harm;
	__shared__ int idx;
	__shared__ int dimj;
	__shared__ int iconta;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x;
	double9 YS = { 0 };
	cuDoubleComplex res[3] = { {0, 0}, {0, 0}, {0, 0} };
	if (tid == 0) {
		n_harm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		pow_rp = make_double3(pow(rp[i], lp - 1), pow(rp[i], lp), pow(rp[i], lp + 1));
		dimj = 3 * n_harm * Np;
		idx = 3 * (blockIdx.y + i * n_harm) * dimj;
		iconta = 3 * (blockIdx.y + i * n_harm);
		res[0] = lp * (lp + 1.0) / pow_rp.y * x[iconta + 2];
		res[1] = cuCadd(lp / pow(rp[i], lp + 2) * x[iconta], -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y * x[iconta + 2]);
		res[2] = 1.0 / pow_rp.z * x[iconta + 1];

		for (int tmpi = 0; tmpi < 6; tmpi++) {
			tmp[tmpi] = make_cuDoubleComplex(0, 0);
		}
	}
	__syncthreads();
	if (tid < n_harm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int ys_id = mp + lp + (lp - 1) * (lp + 1);
		ys_id *= n_harm;
		ys_id += m + l + (l - 1) * (l + 1);
		YS = d_YS[ys_id];
	}
	for (int j = 0; j < Np; j++) {
		if (i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int jconta = 3 * (tid + j * n_harm);
			double plm0 = legendre(l + lp - 2, abs(m - mp), xyz_polar.y);
			double plm1 = legendre(l + lp - 1, abs(m - mp), xyz_polar.y);
			double plm2 = legendre(l + lp, abs(m - mp), xyz_polar.y);

			double rij0 = pow(xyz_polar.x, -l - lp - 1.0);
			double rij1 = pow(xyz_polar.x, -l - lp);
			double rij2 = pow(xyz_polar.x, -l - lp + 1.0);

			cuDoubleComplex exp_phi = cuexp(0, (m - mp) * xyz_polar.z);

			cuDoubleComplex tmp0 = x[jconta];
			cuDoubleComplex tmp1 = x[jconta + 1];
			cuDoubleComplex tmp2 = x[jconta + 2];
			tmp0 = cuCmul(tmp0, exp_phi);
			tmp1 = cuCmul(tmp1, exp_phi);
			tmp2 = cuCmul(tmp2, exp_phi);

			double coeff = YS.ki * pow_rp.x * plm2 * rij0;
			res[0] = cuCadd(res[0], coeff * tmp0);

			coeff = YS.ji * pow_rp.x * plm1 * rij1;
			res[0] = cuCadd(res[0], coeff * make_cuDoubleComplex(-tmp1.y, tmp1.x));

			coeff = (YS.ii * pow_rp.x * plm2 + YS.kk * pow_rp.x * plm0) * rij2 + YS.ik * pow_rp.z * rij0 * plm2;
			res[0] = cuCadd(res[0], coeff * tmp2);

			coeff = YS.jk * pow_rp.z * rij0 * plm2;
			res[1] = cuCadd(res[1], coeff * tmp2);

			coeff = YS.jj * pow_rp.y * plm2 * rij0;
			res[2] = cuCadd(res[2], coeff * tmp1);

			coeff = YS.ij * pow_rp.y * plm1 * rij1;
			res[2] = cuCadd(res[2], coeff * make_cuDoubleComplex(-tmp2.y, tmp2.x));
		}
	}

	int size_warp = (tid >= 32) ? n_harm - 31 : 32;
	res[0].x = warpReduceSum(res[0].x, size_warp);
	res[0].y = warpReduceSum(res[0].y, size_warp);
	res[1].x = warpReduceSum(res[1].x, size_warp);
	res[1].y = warpReduceSum(res[1].y, size_warp);
	res[2].x = warpReduceSum(res[2].x, size_warp);
	res[2].y = warpReduceSum(res[2].y, size_warp);
	if (tid % warpSize == 0) {
		tmp[tid / warpSize * 3] = res[0];
		tmp[tid / warpSize * 3 + 1] = res[1];
		tmp[tid / warpSize * 3 + 2] = res[2];
	}
	__syncthreads();
	if (tid == 0) {
		y[iconta] = cuCadd(beta * y[iconta], alpha * cuCadd(tmp[0], tmp[3]));
		y[iconta + 1] = cuCadd(beta * y[iconta + 1], alpha * cuCadd(tmp[1], tmp[4]));
		y[iconta + 2] = cuCadd(beta * y[iconta + 2], alpha * cuCadd(tmp[2], tmp[5]));
	}
}
__global__ void _compute_yb_slip_cuda_F(cuDoubleComplex* y, cuDoubleComplex* Be, cuDoubleComplex* g, int Np, double* d_coefC_pre, double* d_cnlm, double3* xyz, double* rp, double* zetap) {
	__shared__ cuDoubleComplex tmp[2];
	__shared__ double3 pow_rp;
	__shared__ int lp;
	__shared__ int mp;
	__shared__ int nharm;
	__shared__ int idx;
	__shared__ double val;
	int tid = threadIdx.x;
	int l = 1;
	int m = 0;
	int i = blockIdx.x;
	double _pre = 0;
	cuDoubleComplex res = {0, 0};
	if (tid == 0) {
		nharm = gridDim.y;
		lp = lm2idx_gpu[2 * blockIdx.y];
		mp = lm2idx_gpu[2 * blockIdx.y + 1];
		pow_rp = make_double3(pow(rp[i], lp - 1), pow(rp[i], lp), pow(rp[i], lp + 1));
		int idy = (blockIdx.y + i * nharm);
		idx = 3 * idy;
		val = double(lp * (lp + 1)) * d_cnlm[blockIdx.y];
		res = zetap[i] * val * cuCadd(pow(rp[i], lp - 1.0) * Be[idy], pow(rp[i], -lp - 2.0) * g[idy]);
		for (int tmpi = 0; tmpi < 2; tmpi++) {
			tmp[tmpi] = make_cuDoubleComplex(0, 0);
		}
	}
	__syncthreads();
	if (tid < nharm) {
		l = lm2idx_gpu[2 * tid];
		m = lm2idx_gpu[2 * tid + 1];
		int pre_id = mp + lp + (lp - 1) * (lp + 1);
		pre_id *= nharm;
		pre_id += m + l + (l - 1) * (l + 1);
		_pre = d_coefC_pre[pre_id];
	}
	for (int j = 0; j < Np; j++) {
		if (i != j) {
			double3 xyz_polar = xyz[i * Np + j];
			int idy = tid + j * nharm;
			double plm = legendre(l + lp, abs(m - mp), xyz_polar.y);
			double rij = pow(xyz_polar.x, -l - lp - 1.0);
			
			cuDoubleComplex Be_coeff = zetap[i] * pow(rp[i], lp - 1.0) * val * _pre * plm * rij * cuexp(0, (m - mp) * xyz_polar.z);
			Be_coeff = cuCmul(Be_coeff, g[idy]);
			res = cuCadd(res, Be_coeff);
		}
	}
	int size_warp = (tid >= 32) ? nharm - 31 : 32;
	res.x = warpReduceSum(res.x, size_warp);
	res.y = warpReduceSum(res.y, size_warp);
	if (tid % warpSize == 0) {
		tmp[tid / warpSize] = res;
	}
	__syncthreads();
	if (tid == 0) {
		y[idx] = cuCadd(tmp[0], tmp[1]);
		y[idx + 1] = y[idx];
	}
}

Hydrodynamic::Hydrodynamic(BeadsModel& _model) : Matrixfree(_model) {
	cols = 9;
	nharm = Nc * (Nc + 2) * 3;
	dim = Np * nharm;
	int harm_num = Nc * (Nc + 2);
	//build clm and factori function in array;
	std::vector<double> _facto = std::vector<double>(5 * (Nc + 1), -1);
	_cnlm = std::vector<std::vector<double>>(3 * Nc, std::vector<double>(6 * Nc + 1, 1));
	std::vector<double> cnlm_cuda = std::vector<double>((Nc + 2) * Nc + 1, 1);
	offset_cnlm = 3 * Nc;
	_facto[0] = 1.0;
	for (int i = 1; i < _facto.size(); i++) {
		_facto[i] = std::tgamma(i + 1);
	}
	double pi4 = 4.0 * 3.1415926535897932384626433;
	for (int l = 0; l < 3 * Nc; l++) {
		for (int m = -3 * int(Nc); m <= 3 * int(Nc); m++) {
			int lm_pos = m + l;
			int lm_neg = l - m;
			if (lm_pos < 0 || lm_neg < 0) {
				lm_pos = abs(l + abs(m));
				lm_neg = abs(l - abs(m));
			}
			_cnlm[l][m + offset_cnlm] = sqrt(pi4 / (2.0 * l + 1.0) * _facto[lm_pos] / _facto[lm_neg]);
			if (l >= 1 && l <= Nc && abs(m) <= l) {
				int id = l * l + l + m - 1;
				cnlm_cuda[id] = _cnlm[l][m + offset_cnlm];
			}
		}
	}
	
	cudaMalloc(&d_cnlm, sizeof(double) * cnlm_cuda.size());
	cudaMemcpy((void*)d_cnlm, (void*)cnlm_cuda.data(), sizeof(double) * cnlm_cuda.size(), cudaMemcpyHostToDevice);

	_coefC_pre = std::vector<std::vector<double>>(harm_num, std::vector<double>(harm_num, 0));
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
	//build the pre calculted coeff include YS, coeff for Mc, and cnlm
	std::vector<std::vector<double9>> _coeff_YS = std::vector<std::vector<double9>>(harm_num, std::vector<double9>(harm_num, { 0,0,0,0,0,0,0,0,0 }));
	for (int lp = 1; lp <= Nc; lp++) {
		for (int mp = -lp; mp <= lp; mp++) {
			int i = mp + lp + (lp - 1) * (lp + 1);
			for (int l = 1; l <= Nc; l++) {
				for (int m = -l; m <= l; m++) {
					int j = m + l + (l - 1) * (l + 1);
					double pre = _cnlm[lp][mp + offset_cnlm] / _cnlm[l][m + offset_cnlm];
					double s11 = pow(-1, lp + mp + 1.0) / (l + 1.0) / (2 * l + 1.0) / (lp + 1.0) *
						_facto[l + lp - m + mp] / _facto[l - m] / _facto[lp + mp];

					_coeff_YS[i][j].jj = s11 * _cnlm[l + lp][m - mp + offset_cnlm] * pre * lp * (lp + 1.0) * (2.0 * l + 1.0);

					_coeff_YS[i][j].ik = -s11 * (l + 1) * lp / (2 * lp + 1) / (2 * lp + 3) * _cnlm[l + lp][m - mp + offset_cnlm] *
						pre * (lp + 1.0) * (2.0 * lp + 1.0) * (2.0 * lp + 3.0) / 2.0 * (2.0 * l + 1.0) * l * (2.0 * l - 1.0);

					_coeff_YS[i][j].jk = -s11 * (l + 1) * lp / (2 * lp + 1) / (2 * lp + 3) * _cnlm[l + lp][m - mp + offset_cnlm] *
						pre * (lp + 1.0) * (2.0 * lp + 1.0) * (2.0 * l + 1.0) * l * (2.0 * l - 1.0);

					_coeff_YS[i][j].ij = (lp + l - m + mp == 0) ? 0 : -s11 * (l + 1) * (mp * l + m * lp) / l / lp / (lp + l - m + mp) * _cnlm[l + lp - 1][m - mp + offset_cnlm] *
						pre * (lp + 1.0) * lp * (2.0 * l + 1.0) * l * (2.0 * l - 1.0);

					_coeff_YS[i][j].ki = -s11 * l * (lp + 1) / (2 * l + 1) / (2 * l + 3) * _cnlm[l + lp][m - mp + offset_cnlm] *
						pre * lp * (2.0 * lp + 1.0) * (2.0 * l + 1.0) * (2.0 * l + 3.0);

					_coeff_YS[i][j].ji = (lp + l - m + mp == 0) ? 0 : -s11 * (lp + 1) * (mp * l + m * lp) / l / lp / (lp + l - m + mp) * _cnlm[l + lp - 1][m - mp + offset_cnlm] *
						pre * lp * (2.0 * lp + 1.0) * (2.0 * l + 1.0);

					_coeff_YS[i][j].kk = ((l + lp - m + mp) == 0 || (l + lp - m + mp - 1) == 0) ? 0 : s11 * (l + 1) * (lp + 1) * (l * lp - 2 * l - 2 * lp + 1) /
						l / lp / (2 * l - 1) / (2 * lp - 1) / (2 * l + 2 * lp - 1) / (l + lp - m + mp) / (l + lp - m + mp - 1) * (-l * lp * (l + lp) + 2 * mp
							* mp * l * l + 2 * m * m * lp * lp + (4 * m * mp + 1) * l * lp - mp * (2 * m + mp) * l - m * (2 * mp + m) * lp + m * mp) *
						_cnlm[l + lp - 2][m - mp + offset_cnlm] * pre * lp * (2.0 * lp + 1.0) * (2.0 * l + 1.0) * l * (2.0 * l - 1.0);

					_coeff_YS[i][j].ii = 0.50 * (l + 1) * (lp + 1) / (2 * l + 2 * lp - 1) * s11 * _cnlm[l + lp][m - mp + offset_cnlm]
						* pre * lp * (2.0 * lp + 1.0) * (2.0 * l + 1.0) * l * (2.0 * l - 1.0);

					// add the ynm pre_term 
					if (m - mp > 0) {
						_coeff_YS[i][j].ki *= 1.0 / _cnlm[l + lp][m - mp + offset_cnlm];
						_coeff_YS[i][j].ji *= 1.0 / _cnlm[l + lp - 1][m - mp + offset_cnlm];
						_coeff_YS[i][j].ii *= 1.0 / _cnlm[l + lp][m - mp + offset_cnlm];
						_coeff_YS[i][j].kk *= 1.0 / _cnlm[l + lp - 2][m - mp + offset_cnlm];
						_coeff_YS[i][j].ik *= 1.0 / _cnlm[l + lp][m - mp + offset_cnlm];
						_coeff_YS[i][j].jk *= 1.0 / _cnlm[l + lp][m - mp + offset_cnlm];
						_coeff_YS[i][j].jj *= 1.0 / _cnlm[l + lp][m - mp + offset_cnlm];
						_coeff_YS[i][j].ij *= 1.0 / _cnlm[l + lp - 1][m - mp + offset_cnlm];
					}
					else {
						double value = (abs(m - mp) % 2 == 0) ? 1.0 : -1.0;
						_coeff_YS[i][j].ki *= 1.0 / _cnlm[l + lp][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].ji *= 1.0 / _cnlm[l + lp - 1][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].ii *= 1.0 / _cnlm[l + lp][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].kk *= 1.0 / _cnlm[l + lp - 2][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].ik *= 1.0 / _cnlm[l + lp][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].jk *= 1.0 / _cnlm[l + lp][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].jj *= 1.0 / _cnlm[l + lp][mp - m + offset_cnlm] * value;
						_coeff_YS[i][j].ij *= 1.0 / _cnlm[l + lp - 1][mp - m + offset_cnlm] * value;
					}
				}
			}
		}
	}
	cudaMalloc(&d_YS, sizeof(double9) * harm_num * harm_num);
	for (int i = 0; i < harm_num; i++) {
		double9* ptr = _coeff_YS[i].data();
		double9* dstptr = d_YS + i * harm_num;
		cudaMemcpy((void*)dstptr, (void*)ptr, sizeof(double9) * harm_num, cudaMemcpyHostToDevice);
	}

	cudaMalloc(&d_coefC_pre, sizeof(double)* harm_num * harm_num);
	for (int i = 0; i < harm_num; i++)
		cudaMemcpy((void*)(d_coefC_pre + i * harm_num), (void*)_coefC_pre[i].data(), sizeof(double) * harm_num, cudaMemcpyHostToDevice);

	std::vector<int> lm2idx(2 * Nc * (Nc + 2), 0);
	for (int l = 1; l <= Nc; l++) {
		for (int m = -l; m <= l; m++) {
			int idx = m + l + l * l - 1;
			lm2idx[idx * 2] = l;
			lm2idx[idx * 2 + 1] = m;
		}
	}
	cudaMemcpyToSymbol(lm2idx_gpu, lm2idx.data(), lm2idx.size() * sizeof(int), 0, cudaMemcpyHostToDevice);
}

void Hydrodynamic::set_slip_condition(std::vector<std::complex<double>>& Be, std::vector<std::complex<double>>& g) {
	yb = std::vector<std::complex<double>>(dim * cols, 0);
	std::vector<std::vector<double>> ux(Np, std::vector<double>(6, 0.0));
	std::vector<std::vector<double>> uy(Np, std::vector<double>(6, 0.0));
	std::vector<std::vector<double>> uz(Np, std::vector<double>(6, 0.0));
	std::vector<std::vector<double>> wx(Np, std::vector<double>(6, 0.0));
	std::vector<std::vector<double>> wy(Np, std::vector<double>(6, 0.0));
	std::vector<std::vector<double>> wz(Np, std::vector<double>(6, 0.0));
	std::vector<double> up;
	int dimj = nharm * Np;
	for (int i = 0; i < Np; i++) {
		ux[i][0] = 1.0;
		uy[i][1] = 1.0;
		uz[i][2] = 1.0;
		wx[i][3] = 1.0;
		wy[i][4] = 1.0;
		wz[i][5] = 1.0;
		std::vector<double> point = { model.xyz[i * 3] - model.center[0],
			model.xyz[i * 3 + 1] - model.center[1],
			model.xyz[i * 3 + 2] - model.center[2] };
		up = crossp(wx[i][3], wx[i][4], wx[i][5], point);
		ux[i][3] += up[0];
		uy[i][3] += up[1];
		uz[i][3] += up[2];

		up = crossp(wy[i][3], wy[i][4], wy[i][5], point);
		ux[i][4] += up[0];
		uy[i][4] += up[1];
		uz[i][4] += up[2];

		up = crossp(wz[i][3], wz[i][4], wz[i][5], point);
		ux[i][5] += up[0];
		uy[i][5] += up[1];
		uz[i][5] += up[2];
	}
	double cnlm_1_1 = 2.8944050182330705;
	double cnlm_1_0 = 2.0466534158929770;
	std::complex<double> imag_1 = std::complex<double>(0, 1);

	for (int j = 0; j < 6; j++) {
		for (int i = 0; i < Np; i++) {
			for (int m = -1; m <= 1; m++) {
				double kdelta_neg = kdelta(m, -1) - kdelta(m, 1);
				double kdelta_pos = kdelta(m, -1) + kdelta(m, 1);
				size_t idx = (3 * (m + 1) + i * nharm);
				yb[idx + j * dimj] = 3.0 / 2.0 * (cnlm_1_1 * kdelta_neg * ux[i][j] + imag_1 * cnlm_1_1 * kdelta_pos * uy[i][j] + 2.0 * cnlm_1_0 * kdelta(m, 0) * uz[i][j]);
				yb[idx + 1 + j * dimj] = std::complex<double>(0, 0);
				yb[idx + 2 + j * dimj] = model.rp[i] * (cnlm_1_1 * kdelta_neg * wx[i][j] + imag_1 * cnlm_1_1 * kdelta_pos * wy[i][j] + 2.0 * cnlm_1_0 * kdelta(m, 0) * wz[i][j]);
			}
		}
	}
	
	
	cuDoubleComplex* d_Be;
	cuDoubleComplex* d_g;
	cuDoubleComplex* d_yb;
	cudaMalloc(&d_Be, Be.size() * sizeof(cuDoubleComplex));
	cudaMalloc(&d_g, g.size() * sizeof(cuDoubleComplex));
	cudaMalloc(&d_yb, dim * 3 * sizeof(cuDoubleComplex));
	
	cudaMemcpy((void*)d_Be, (void*)Be.data(), sizeof(cuDoubleComplex) * Be.size(), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_g, (void*)g.data(), sizeof(cuDoubleComplex) * g.size(), cudaMemcpyHostToDevice);

	dim3 blockDims(Nc * (Nc + 2), 1);
	dim3 gridDims(Np, Nc * (Nc + 2));
	cudaMemset(d_yb, 0, dim * 3 * sizeof(cuDoubleComplex));
	int dim_Be = Np * Nc * (Nc + 2);
	
	for (int i = 0; i < 3; i++) {
		_compute_yb_slip_cuda_F << <gridDims, blockDims >> > (d_yb + i * dim, d_Be + i * dim_Be, d_g + i * dim_Be,
			Np, d_coefC_pre, d_cnlm, model.d_xyz, model.d_rp, model.d_zetap);
	}
	cudaMemcpy((void*)(yb.data() + 6 * dim), (void*)d_yb, sizeof(cuDoubleComplex) * 3 * dim, cudaMemcpyDeviceToHost);
	cudaFree(d_Be);
	cudaFree(d_g);
	cudaFree(d_yb);
}

void Hydrodynamic::right_side_F(cuDoubleComplex* y, int icol, int jcol) const {
	if (yb.size() != cols * dim)
		throw std::runtime_error("Error: right hand vector not initialize");
	int dimj = nharm * (jcol - icol);
	int offset = icol * nharm;
	cudaMemcpy2D((void*)y, dimj * sizeof(cuDoubleComplex), 
		(void*)(yb.data() + offset), dim * sizeof(cuDoubleComplex),
		dimj * sizeof(cuDoubleComplex), cols, cudaMemcpyHostToDevice);
	return;
} 

void Hydrodynamic::right_side_F_mkl(MKL_Complex16* y, int icol, int jcol) const {
	if (yb.size() != cols * dim)
		throw std::runtime_error("Error: right hand vector not initialize");
	int dimj = nharm * (jcol - icol);
	int offset = icol * nharm;
	mkl_zomatcopy('C', 'N', dimj, cols, { 1.0, 0.0 },
		reinterpret_cast<const MKL_Complex16*>(yb.data() + offset), dim,
		y, dimj);
	return;
}

void Hydrodynamic::compute_matrix(cuDoubleComplex* Mc, int icol, int jcol, int irow, int jrow, bool c_major) const {
	int N = Nc * (Nc + 2);
	dim3 blockDims(N, 1);
	dim3 gridDims(jcol - icol, N);
	int dimx = (jcol - icol) * nharm;
	int dimy = (jrow - irow) * nharm;
	cudaMemset(Mc, 0, dimx * dimy * sizeof(cuDoubleComplex));
	if (c_major)
		_compute_matrix_cuda_C << <gridDims, blockDims >> > (Mc, icol, irow, jrow, Np, d_YS, model.d_xyz, model.d_rp);
	else 
		_compute_matrix_cuda_F << <gridDims, blockDims >> > (Mc, icol, irow, jrow, Np, d_YS, model.d_xyz, model.d_rp);
}

void Hydrodynamic::get_matrix(std::vector<std::complex<double>>& Mcc, bool c_major) const {
	int subi0 = 2;
	int subj0 = 25;
	int N = Nc * (Nc + 2);
	int subsize = subi0 * subj0 * 9 * N * N;
	cuDoubleComplex* Mc;
	cudaMalloc(&Mc, subsize * sizeof(cuDoubleComplex));
	cuDoubleComplex* Mc2;
	cudaMalloc(&Mc2, dim * dim * sizeof(cuDoubleComplex));
	int subi = subi0;
	int subj = subj0;
	cudaMemset(Mc2, 0, dim * dim * sizeof(cuDoubleComplex));
	for (int i = 0; i < Np; i += subi0) {
		subi = subi0;
		if (i + subi0 > Np)
			subi = Np - i;
		for (int j = 0; j < Np; j += subj0) {
			subj = subj0;
			if (j + subj0 > Np)
				subj = Np - j;

			int preid = (j + i * Np) * (N * N * 9);
			int dimj = subj * 3 * N;
			compute_matrix(Mc, i, i + subi, j, j + subj, c_major);
			cudaMemcpy2D(Mc2 + j * 3 * N + i * 3 * N * dim, dim * sizeof(cuDoubleComplex),
				Mc, dimj * sizeof(cuDoubleComplex),
				dimj * sizeof(cuDoubleComplex), N * 3 * subi,
				cudaMemcpyDeviceToDevice);
		}
	}
	cudaMemcpy(Mcc.data(), Mc2, dim * dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

void Hydrodynamic::matmult(cuDoubleComplex* x, double alpha, cuDoubleComplex* y, double beta) const {
	int num_thread = 32;
	int harm_num = Nc * (Nc + 2);
	if (harm_num > 32)
		num_thread = 64;
	dim3 blockDims(num_thread, 1);
	dim3 gridDims(Np, harm_num);
	for (int i = 0; i < cols; i++) {
		_matmult_cuda_F << <gridDims, blockDims >> > (x + i * dim, alpha, y + i * dim, beta, Np, d_YS, model.d_xyz, model.d_rp);
	}
	/*
	cuDoubleComplex alpha1 = make_cuDoubleComplex(1.0, 0.0);
	cuDoubleComplex beta1 = make_cuDoubleComplex(0.0, 0.0);
	cublasHandle_t handle = nullptr;
	cublasCreate(&handle);
	cuDoubleComplex* d_Mc = nullptr;
	cudaMalloc(&d_Mc, sizeof(cuDoubleComplex) * nharm * dim);
	cuDoubleComplex* d_tmp = nullptr;
	cudaMalloc(&d_tmp, sizeof(cuDoubleComplex) * nharm * cols);
	cuDoubleComplex* d_tmp2 = nullptr;
	cudaMalloc(&d_tmp2, sizeof(cuDoubleComplex) * dim * cols);
	for (int i = 0; i < Np; i++) {
		int offset = i * nharm;
		compute_matrix(d_Mc, i, i + 1, 0, Np, false);
		cublasZgemm_v2(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			nharm, cols, dim,
			&alpha1,
			d_Mc, nharm,
			x, dim,
			&beta1,
			d_tmp, nharm);
		cudaMemcpy2D(d_tmp2 + offset, dim * sizeof(cuDoubleComplex),
			d_tmp, nharm * sizeof(cuDoubleComplex),
			nharm * sizeof(cuDoubleComplex), cols,
			cudaMemcpyDeviceToDevice);
		//transpose(d_y_tmp + offset, yy + offset, m, cols);
	}
	cuDoubleComplex gamma = make_cuDoubleComplex(-1.0, 0.0);
	cublasZaxpy(handle,
		dim * cols,
		&gamma,
		d_tmp2,
		1,
		y, 1);
	*/
}