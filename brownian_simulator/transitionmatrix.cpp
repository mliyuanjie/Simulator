#include "transitionmatrix.h" 
#include <mkl_vsl.h>
#include <random>
#include <mkl.h>
#include <mkl_cblas.h>

constexpr double pi = 3.1415926535897932384626433;
void emit(double* mu, double o0, double sigma0, double* prob, int n) {
	double norm_const = 1.0 / std::sqrt(2.0 * pi) / sigma0;
	double temp = -1.0 / (2 * sigma0 * sigma0);
	std::fill(prob, prob + n, -o0);
	vdAdd(n, mu, prob, prob);
	vdMul(n, prob, prob, prob);
	cblas_dscal(n, temp, prob, 1);
	vdExp(n, prob, prob);
	cblas_dscal(n, norm_const, prob, 1);
	return;
}

double normalize(double* p, int n) {
	double prob = 0;
	for (int i = 0; i < n; i++) {
		prob += p[i];
	}
	if (prob == 0) {
		return 0;
	}
	for (int i = 0; i < n; i++) {
		p[i] /= prob;
	}
	return prob;
}

double compute_transition_matrix(std::vector<float>& cura, double sigma, int skip,
	std::vector<float>& curb, double curmin) {
	VSLStreamStatePtr stream;
	std::random_device device;
	std::mt19937 rd{ device() };
	unsigned long seed = rd();
	vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);
	std::vector<float> noise(curb.size(), 0);
	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, curb.size(), noise.data(), 0.0, sigma);
	vsAdd(curb.size(), curb.data(), noise.data(), curb.data());

	double min = DBL_MAX;
	double max = -DBL_MAX;
	for (int i = 0; i < cura.size(); i++) {
		min = (min <= cura[i]) ? min : cura[i];
		max = (max >= cura[i]) ? max : cura[i];
	}
	double min_max = max - min;
	int n = ceil((min_max) / curmin);
	std::vector<double> matrix(n * n, 0);
	std::vector<double> matrix_sum(n, 0);
	for (int k = 1; k < cura.size(); k++) {
		int i = ceil((cura[k - 1] - min) / curmin) - 1;
		int j = ceil((cura[k] - min) / curmin) - 1;
		i = (i < 0) ? 0 : i;
		j = (j < 0) ? 0 : j;
		matrix[i * n + j]++;
		matrix_sum[i]++;
	}
	std::vector<double> A(n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i * n + j] = (i == j) ? 1 : 0;
			if (matrix_sum[i] != 0)
				matrix[i * n + j] /= matrix_sum[i];
		}
	}
	std::vector<double> prob0(n, 0);
	std::vector<double> state(n, 0);
	for (int i = 0; i < cura.size(); i++) {
		int id = ceil((cura[i] - min) / curmin) - 1;
		id = (id < 0) ? 0 : id;
		prob0[id]++;
	}
	state[0] = curmin / 2 + min;
	for (int i = 1; i < n; i++) {
		state[i] = state[i - 1] + curmin;
	}
	cblas_dscal(n, 1.0 / double(cura.size()), prob0.data(), 1);
	std::vector<double> temp(n * n);
	int power = skip;
	while (power > 0) {
		if (power % 2 == 1) {
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				n, n, n, 1.0, A.data(), n, matrix.data(), n,
				0, temp.data(), n);
			memcpy(A.data(), temp.data(), n * n * sizeof(double));
		}
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			n, n, n, 1.0, matrix.data(), n, matrix.data(), n,
			0, temp.data(), n);
		memcpy(matrix.data(), temp.data(), n * n * sizeof(double));
		power /= 2;
	}
	temp.clear();
	matrix.clear();
	//HMM
	double* alpha_pre = new double[n];
	double* alpha = new double[n];
	emit(state.data(), curb[0], sigma, alpha, n);
	vdMul(n, alpha, prob0.data(), alpha);
	std::vector<double> scale(curb.size(), 0);
	scale[0] = normalize(alpha, n);
	for (int i = 1; i < curb.size(); i++) {
		memcpy(alpha_pre, alpha, n * sizeof(double));
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, 1, n, 1.0,
			A.data(), n, alpha_pre, 1, 0.0, alpha, 1);
		emit(state.data(), curb[i], sigma, alpha_pre, n);
		vdMul(n, alpha, alpha_pre, alpha);
		scale[i] = normalize(alpha, n);
	}
	double prob = 0;
	for (int i = 0; i < n; i++) {
		prob += alpha[i];
	}
	prob = -log10(prob);
	for (int i = 0; i < curb.size(); i++) {
		if (scale[i] != 0)
			prob -= log10(scale[i]);
		else
			prob -= DBL_MAX;
	}
	delete[] alpha;
	delete[] alpha_pre;
	return prob;
}

double compute_mean_std(std::vector<float>& cura, double sigma, std::vector<float>& curb) {
	double mean = 0;
	double m2 = 0;
	for (int i = 0; i < cura.size(); i++) {
		double d1 = (cura[i] - mean);
		mean += d1 / double(i + 1);
		double d2 = (cura[i] - mean);
		m2 += d1 * d2;
	}
	double std = sqrt(m2 / double(cura.size() - 1) + sigma * sigma);

	VSLStreamStatePtr stream;
	std::random_device device;
	std::mt19937 rd{ device() };
	unsigned long seed = rd();
	vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);
	std::vector<float> noise(curb.size(), 0);
	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, curb.size(), noise.data(), 0.0, sigma);
	vsAdd(curb.size(), curb.data(), noise.data(), curb.data());
	double meanb = 0;
	double m2b = 0;
	for (int i = 0; i < curb.size(); i++) {
		double d1 = (curb[i] - meanb);
		meanb += d1 / double(i + 1);
		double d2 = (curb[i] - meanb);
		m2b += d1 * d2;
	}
	double stdb = sqrt(m2b / double(curb.size() - 1));
	double distance = abs(meanb - mean) /sqrt(stdb * stdb / curb.size() + std * std / cura.size());
	return distance;
}
