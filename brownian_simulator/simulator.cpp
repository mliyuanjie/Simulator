#include "simulator.h" 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "accumulator.h"
#include "computetensor.h"


void cross_batch(const double* __restrict A, int lda, const double* __restrict B, int ldb, double* __restrict C, int ldc, int n) {
	for (int i = 0; i < n; i++) {
		int ida = i * lda;
		int idb = i * ldb;
		int idc = i * ldc;
		C[idc] = A[ida + 1] * B[idb + 2] - A[ida + 2] * B[idb + 1];
		C[idc + 1] = A[ida + 2] * B[idb] - A[ida] * B[idb + 2];
		C[idc + 2] = A[ida] * B[idb + 1] - A[ida + 1] * B[idb];
	}
}

void axpy_batch(int n, double alpha, const double* x, int lda, double* y, int ldb, int batch) {
	for (int i = 0; i < n; i++) {
		cblas_daxpy(batch, alpha, x + i, lda, y + i, ldb);
	}
}

void dot_batch(const double* __restrict x, int lda, const double* __restrict y, int ldb, int batch, double* __restrict res) {
	for (int i = 0; i < batch; i++) {
		int idx = i * lda;
		int idy = i * ldb;
		res[i] = x[idx] * y[idy] + x[idx + 1] * y[idy + 1] + x[idx + 2] * y[idy + 2];
	}
}

void dot_batch_same(const double* __restrict x, int lda, int batch, double* __restrict res) {
	for (int i = 0; i < batch; i++) {
		int idx = i * lda;
		res[i] = x[idx] * x[idx] + x[idx + 1] * x[idx + 1] + x[idx + 2] * x[idx + 2];
	}
}

void scale_batch(int n, const double* __restrict alpha, const double* x, int lda, double* __restrict y, int ldb, int batch) {
	for (int i = 0; i < batch; i++) {
		int idx = i * lda;
		int idy = i * ldb;
		for (int j = 0; j < 3; j++) {
			y[idy + j] = alpha[i] * x[idx + j];
		}
	}
}

void Simulator::calculate_force_batch(double** d_dx, double** d_force, double** d_t1,
	const double** d_sqm, const double** d_mob, const double** d_efm, const double** d_dipole,
	const double** d_t1_const, const double** d_force_const, const double** d_E, const double** d_dE, int batch) {
	CBLAS_TRANSPOSE transn = CblasNoTrans;
	int ld6 = 6;
	int ld3 = 3;
	int ld1 = 1;
	int ld9 = 9;
	int m6 = 6;
	int m3 = 3;
	int m1 = 1;
	int group_size[1] = { batch };
	//update tensor

	double alpha = co;
	double beta = 0.0;
	cblas_dgemm_batch(CblasColMajor, &transn, &transn, &m6, &m1, &m6, &alpha, d_sqm, &ld6, d_force_const, &ld6,
		&beta, d_dx, &ld6, 1, group_size);
	//update electrophoretic
	alpha = pa;
	beta = 1.0;
	cblas_dgemm_batch(CblasColMajor, &transn, &transn, &m6, &m1, &m3, &alpha, d_efm, &ld6, d_E, &ld3,
		&beta, d_dx, &ld6, 1, group_size);
	//printf("dx0: %f, dx1: %f, dx2: %f, dx3: %f, dx4: %f, dx5: %f", 
	//	d_dx[0][0], d_dx[0][1], d_dx[0][2], d_dx[0][3], d_dx[0][4], d_dx[0][5]);
	//t1 = p * E
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemm_batch(CblasColMajor, &transn, &transn, &m3, &m1, &m3, &alpha, d_dipole, &ld3, d_E, &ld3,
		&beta, d_t1, &ld3, 1, group_size);
	//a1 = t1 * dE, Ft = d_rng
	cblas_dgemm_batch(CblasColMajor, &transn, &transn, &m1, &m3, &m3, &alpha, d_t1_const, &ld1, d_dE, &ld3,
		&beta, d_force, &ld1, 1, group_size);
	//a2 = t1 x E
	cross_batch(d_t1[0], ld3, d_E[0], ld3, d_force[0] + 3, ld6, batch);
	//update torque force 
	alpha = 1.0;
	beta = 1.0;
	
	cblas_dgemm_batch(CblasColMajor, &transn, &transn, &m6, &m1, &m6, &alpha, d_mob, &ld6, d_force_const, &ld6,
		&beta, d_dx, &ld6, 1, group_size);
}

void Simulator::calculate_efield_batch(double* E, double* dE, int batch) {
	memset(E, 0, 3 * batch * sizeof(double));
	memset(dE, 0, 9 * batch * sizeof(double));
	for (int i = 0; i < batch; i++) {
		E[i * 3 + 2] = 2e6;
	}
}

void Simulator::set_position_batch(double* u, double* xyz, int batch) {
	for (int i = 0; i < batch; i++) {
		memcpy(u + i * 9, u0, 9 * sizeof(double));
		memcpy(xyz + i * 3, xyz0, 3 * sizeof(double));
	}
}

void Simulator::forward_motion_batch(double* x3, double* u9, int step, int skip, int batch) {
	int overallsize = (16 * 11 + 30 + 36 * 2 + 18) * batch;
	size_t overallsize_t = overallsize * sizeof(double) / 1024 / 1024;
	printf("memory: %f MB", double(overallsize_t));
	this->batch = batch;
	std::random_device device;
	std::mt19937 rd{ device() };

	double* efm = new double[18 * batch];
	double* sqm = new double[36 * batch];
	double* mob = new double[36 * batch];
	double* dipole = new double[9 * batch];
	double* resis = new double[9 * batch];
	double* xyz = new double[3 * batch];
	double* u = new double[9 * batch];

	int ld_tensor[2] = { 6, 3 };
	int ld3[2] = { 3, 3 };
	int group_size[2] = { batch * 10, batch };
	int m3[2] = { 3, 3 };
	CBLAS_TRANSPOSE transn[2] = {CblasNoTrans, CblasNoTrans };
	CBLAS_TRANSPOSE transt[2] = { CblasTrans, CblasTrans };
	double alpha[2] = { 1,1 };
	double beta[2] = { 0,0 };
	const double** d_u = new const double* [batch * 11];
	const double** d_tensor_const = new const double* [batch * 11];
	const double** d_tmp_const = new const double* [batch * 11];
	double** d_tensor = new double* [batch * 11];
	double** d_tmp = new double* [batch * 11];
	double* tmp = new double[batch * 99];
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < 11; j++) {
			d_u[i * 11 + j] = u + i * 9;
			d_tmp[i * 11 + j] = tmp + i * 99 + j * 9;
			d_tmp_const[i * 11 + j] = tmp + i * 99 + j * 9;
		}
		d_tensor_const[i * 10] = mob0;
		d_tensor_const[i * 10 + 1] = mob0 + 3;
		d_tensor_const[i * 10 + 2] = mob0 + 18;
		d_tensor_const[i * 10 + 3] = mob0 + 21;
		d_tensor_const[i * 10 + 4] = sqm0;
		d_tensor_const[i * 10 + 5] = sqm0 + 3;
		d_tensor_const[i * 10 + 6] = sqm0 + 18;
		d_tensor_const[i * 10 + 7] = sqm0 + 21;
		d_tensor_const[i * 10 + 8] = efm0;
		d_tensor_const[i * 10 + 9] = efm0 + 3;
		d_tensor[i * 10] = mob + 36 * i;
		d_tensor[i * 10 + 1] = mob + 3 + 36 * i;
		d_tensor[i * 10 + 2] = mob + 18 + 36 * i;
		d_tensor[i * 10 + 3] = mob + 21 + 36 * i;
		d_tensor[i * 10 + 4] = sqm + 36 * i;
		d_tensor[i * 10 + 5] = sqm + 3 + 36 * i;
		d_tensor[i * 10 + 6] = sqm + 18 + 36 * i;
		d_tensor[i * 10 + 7] = sqm + 21 + 36 * i;
		d_tensor[i * 10 + 8] = efm + 18 * i;
		d_tensor[i * 10 + 9] = efm + 3 + 18 * i;
	}
	for (int i = 0; i < batch; i++) {
		d_tensor_const[i + 10 * batch] = dipole0;
		d_tensor[i + 10 * batch] = dipole + 9 * i;
	}
	
	double** d_dx = new double* [batch];
	double** d_force = new double* [batch];
	double** d_t1 = new double* [batch];
	const double** d_sqm = new const double* [batch];
	const double** d_mob = new const double* [batch];
	const double** d_efm = new const double* [batch];
	const double** d_dipole = new const double* [batch];
	const double** d_t1_const = new const double* [batch];
	const double** d_force_const = new const double* [batch];
	const double** d_E = new const double* [batch];
	const double** d_dE = new const double* [batch];
	double* dx = tmp;//new double[6 * batch];
	double* force = tmp + 6 * batch;
	double* t1 = tmp + 12 * batch;
	double* E = tmp + 15 * batch;
	double* dE = tmp + 18 * batch;
	double* utmp = tmp + 27 * batch;//new double[3 * batch];
	double* uscale = tmp + 30 * batch;//new double[batch];
	for (int i = 0; i < batch; i++) {
		d_dx[i] = dx + i * 6;
		d_force[i] = force + i * 6;
		d_t1[i] = t1 + i * 3;
		d_force_const[i] = force + i * 6;
		d_t1_const[i] = t1 + i * 3;
		d_sqm[i] = sqm + i * 36;
		d_mob[i] = mob + i * 36;
		d_efm[i] = efm + i * 18;
		d_dipole[i] = dipole + i * 9;
		d_E[i] = E + i * 3;
		d_dE[i] = dE + i * 9;
	}

	double svd_vt[9];
	double svd_s[3];
	double svd_u[9];
	double svd_superb[2];

	this->set_position_batch(u, xyz, batch);
	for (int i = 0; i < step; i++) {
		if (i % skip == 0) {
			int idx = i / skip;
			memcpy(x3 + idx * 3 * batch, xyz, 3 * batch * sizeof(double));
			memcpy(u9 + idx * 9 * batch, u, 9 * batch * sizeof(double));
		}
		//update the tensor
		
		cblas_dgemm_batch(CblasColMajor, transn, transn, m3, m3, m3, alpha, d_u, ld3, d_tensor_const, ld_tensor,
			beta, d_tmp, ld3, 2, group_size);
		cblas_dgemm_batch(CblasColMajor, transn, transt, m3, m3, m3, alpha, d_tmp_const, ld3, d_u, ld3,
			beta, d_tensor, ld_tensor, 2, group_size);
		
		//set random force 
		this->calculate_efield_batch(E, dE, batch);
		unsigned long seed = rd();
		vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 6 * batch, force, 0.0, 1.0);
		
		this->calculate_force_batch(d_dx, d_force, d_t1, d_sqm, d_mob, d_efm, d_dipole, d_t1_const,d_force_const, d_E, d_dE, batch);
		
		//update x with dx
		axpy_batch(3, 1.0, dx, 6, xyz, 3, batch);
		for (int j = 0; j < 3; j++) {
			mkl_domatcopy('R', 'N', batch, 3, 1.0, u + j * 3, 9, utmp, 3);
			cross_batch(dx + 3, 6, u + j * 3, 9, t1, 3, batch);
			axpy_batch(3, 1.0, t1, 3, utmp, 3, batch);
			dot_batch_same(dx + 3, 6, batch, uscale);
			
			scale_batch(3, uscale, u + j * 3, 9, t1, 3, batch);
			axpy_batch(3, -0.5, t1, 3, utmp, 3, batch);
			dot_batch(u + j * 3, 9, dx + 3, 6, batch, uscale);
			scale_batch(3, uscale, dx + 3, 6, t1, 3, batch);
			axpy_batch(3, -1.0, t1, 3, utmp, 3, batch);
			mkl_domatcopy('R', 'N', batch, 3, 1.0, utmp, 3, u + j * 3, 9);
		}
		//upate u and tensor
		for (int j = 0; j < batch; j++) {
			LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', 3, 3, u + j * 9, 3, svd_s, svd_u, 3, svd_vt, 3, svd_superb);
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				3, 3, 3,
				1.0, svd_u, 3, svd_vt, 3,
				0.0, u + j * 9, 3);
		}
	}

	delete[] d_u;
	delete[] d_tensor_const;
	delete[] d_tmp_const;
	delete[] d_tensor;
	delete[] d_tmp;
	delete[] d_dx;
	delete[] d_force;
	delete[] d_t1;
	delete[] d_sqm;
	delete[] d_mob;
	delete[] d_efm;
	delete[] d_dipole;
	delete[] d_t1_const;
	delete[] d_force_const;
	delete[] d_E;
	delete[] d_dE;

	delete[] efm;
	delete[] sqm;
	delete[] mob;
	delete[] dipole;
	delete[] resis;
	delete[] xyz;
	delete[] u;
	delete[] tmp;
}

void Simulator::loadfile(std::string& fn_tensor, double dt,
	double temperature, double salt, double salt_resis,
	double pore_radius, double pore_length, double epw) {
	std::ifstream infile(fn_tensor);
	std::string line;
	int matrix_id = 0;
	std::vector<double> resis0_tmp;
	std::vector<double> mob0_tmp;
	std::vector<double> efm0_tmp;
	std::vector<double> dipole0_tmp;
	while (std::getline(infile, line)) {
		if (line.empty() || line[0] == '#') continue;
		else {
			if (matrix_id >= 0 && matrix_id < 3) {
				std::istringstream iss(line);
				double num;
				while (iss >> num) {
					resis0_tmp.push_back(num);
				}
			}
			else if (matrix_id >= 3 && matrix_id < 9) {
				std::istringstream iss(line);
				double num;
				while (iss >> num) {
					mob0_tmp.push_back(num);
				}
			}
			else if (matrix_id >= 9 && matrix_id < 15) {
				std::istringstream iss(line);
				double num;
				while (iss >> num) {
					efm0_tmp.push_back(num);
				}
			}
			else if (matrix_id >= 15 && matrix_id < 18) {
				std::istringstream iss(line);
				double num;
				while (iss >> num) {
					dipole0_tmp.push_back(num);
				}
			}
			matrix_id++;
		}
	}

	if (mob0_tmp.empty()) {
		std::cerr << "error: no tensor computed" << std::endl;
	}
	double kb = 1.380649e-23;
	double pi = 3.14159265358979323846;
	double eps0 = 8.854187817e-12;
	dt *= 1e-12;
	temperature += 273.15;
	double eta = 2.414e-5 * pow(10, (247.8 / (temperature - 140)));
	this->co = sqrt(2 * kb * temperature * dt / (6 * pi * eta));
	double xi = kb * temperature / 1.602176634e-19;
	this->pa = eps0 * epw * xi / eta * dt / this->rp;
	this->pore_radius = pore_radius;
	this->pore_length = pore_length;

	mkl_domatcopy('R', 'T', 6, 6, 1.0 / rp, mob0_tmp.data(), 6, mob0, 6);
	for (int i = 0; i < 6; i++) {
		cblas_dscal(3, 1.0 / rp, mob0 + 3 + i * 6, 1);
	}
	cblas_dscal(18, 1.0 / rp, mob0 + 18, 1);
	mkl_domatcopy('R', 'T', 6, 3, 1.0, efm0_tmp.data(), 3, efm0, 6);
	for (int i = 0; i < 3; i++) {
		cblas_dscal(3, rp, efm0 + i * 6, 1);
	}
	mkl_domatcopy('R', 'T', 3, 3, pow(this->rp, 3) / salt_resis, resis0_tmp.data(), 3, resist0, 3);
	double dipole_alpha = eps0 * epw * pow(this->rp, 3) * dt / eta;
	mkl_domatcopy('R', 'T', 3, 3, dipole_alpha, dipole0_tmp.data(), 3, dipole0, 3);

	double mob0_t[36];
	double eig_w[6];
	memcpy(mob0_t, mob0, 36 * sizeof(double));
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', 6, mob0_t, 6, eig_w);
	for (int j = 0; j < 6; j++) {
		if (eig_w[j] < 0) {
			eig_w[j] = 0;
			continue;
		}
		eig_w[j] = sqrt(eig_w[j]);
	}
	double eig_diag[36];
	for (int j = 0; j < 6; j++) {
		for (int i = 0; i < 6; i++) {
			eig_diag[i + j * 6] = mob0_t[i + j * 6] * eig_w[j];
		}
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 6, 6, 6, 1.0, 
		eig_diag, 6, mob0_t, 6, 0.0, sqm0, 6);
	return;
}

void Simulator::loadtensor(double* mob, double* resist, double* dipole, double* efm, double dt,
	double temperature, double salt, double salt_resis,
	double pore_radius, double pore_length, double epw) {
	this->pore_length = pore_length;
	this->pore_radius = pore_radius;
	memcpy(this->mob0, mob, 36 * sizeof(double));
	memcpy(this->dipole0, dipole, 9 * sizeof(double));
	memcpy(this->efm0, efm, 18 * sizeof(double));
	memcpy(this->resist0, resist, 9 * sizeof(double));
	double kb = 1.380649e-23;
	double pi = 3.14159265358979323846;
	double eps0 = 8.854187817e-12;
	dt *= 1e-12;
	temperature += 273.15;
	double eta = 2.414e-5 * pow(10, (247.8 / (temperature - 140)));
	this->co = sqrt(2 * kb * temperature * dt / (6 * pi * eta));
	double xi = kb * temperature / 1.602176634e-19;
	this->pa = eps0 * epw * xi / eta * dt / this->rp;

	cblas_dscal(36, 1.0 / rp, this->mob0, 1);
	for (int i = 0; i < 6; i++) {
		cblas_dscal(3, 1.0 / rp, this->mob0 + 3 + i * 6, 1);
	}
	cblas_dscal(18, 1.0 / rp, this->mob0 + 18, 1);
	for (int i = 0; i < 3; i++) {
		cblas_dscal(3, rp, this->efm0 + i * 6, 1);
	}
	cblas_dscal(9, pow(this->rp, 3) / salt_resis, this->resist0, 1);
	double dipole_alpha = eps0 * epw * pow(this->rp, 3) * dt / eta;
	cblas_dscal(9, dipole_alpha, this->dipole0, 1);

	double mob0_t[36];
	double eig_w[6];
	memcpy(mob0_t, this->mob0, 36 * sizeof(double));
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', 6, mob0_t, 6, eig_w);
	for (int j = 0; j < 6; j++) {
		if (eig_w[j] < 0) {
			eig_w[j] = 0;
			continue;
		}
		eig_w[j] = sqrt(eig_w[j]);
	}
	double eig_diag[36];
	for (int j = 0; j < 6; j++) {
		for (int i = 0; i < 6; i++) {
			eig_diag[i + j * 6] = mob0_t[i + j * 6] * eig_w[j];
		}
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 6, 6, 6, 1.0,
		eig_diag, 6, mob0_t, 6, 0.0, this->sqm0, 6);
	return;
}

void Simulator::forward_motion(double* x3, double* u9, int step, int skip) {
	e::Map<e::Matrix<double, 6, 6, e::ColMajor>> mat_mob0(mob0);
	e::Map<e::Matrix<double, 6, 6, e::ColMajor>> mat_sqm0(sqm0);
	e::Map<e::Matrix<double, 6, 3, e::ColMajor>> mat_efm0(efm0);
	e::Map<e::Matrix3d> mat_dipole0(dipole0);

	std::random_device device;
	std::mt19937 urng{ device() };
	std::normal_distribution<double> dis{ 0, 1 };
	e::Matrix<double, 6, 6> mob;
	e::Matrix<double, 6, 3> efm;
	e::Matrix3d dipole;
	e::Vector3d p1;
	e::Matrix<double, 6, 6> sqm;
	e::Matrix3d u = e::Map<e::Matrix3d>(u0);
	e::Matrix3d mat_u0 = e::Map<e::Matrix3d>(u0);
	e::Vector<double, 6> dx;
	e::Vector<double, 6> Ft;
	e::Vector<double, 6> randx;
	e::Vector3d E;
	e::Matrix3d dE;
	e::Vector3d xyz = e::Map<e::Vector3d>(xyz0);
	e::Vector3d mat_xyz0 = e::Map<e::Vector3d>(xyz0);
	
	RotationAccumulator accu;

	for (int i = 0; i < step; i++) {
		if (i % skip == 0) {
			int idx = i / skip;
			memcpy(x3 + idx * 3, xyz.data(), 3 * sizeof(double));
			memcpy(u9 + idx * 9, u.data(), 9 * sizeof(double));
		}
		mob.block(0, 0, 3, 3).noalias() = u * mat_mob0.block(0, 0, 3, 3) * u.transpose();
		mob.block(0, 3, 3, 3).noalias() = u * mat_mob0.block(0, 3, 3, 3) * u.transpose();
		mob.block(3, 0, 3, 3).noalias() = u * mat_mob0.block(3, 0, 3, 3) * u.transpose();
		mob.block(3, 3, 3, 3).noalias() = u * mat_mob0.block(3, 3, 3, 3) * u.transpose();
		sqm.block(0, 0, 3, 3).noalias() = u * mat_sqm0.block(0, 0, 3, 3) * u.transpose();
		sqm.block(0, 3, 3, 3).noalias() = u * mat_sqm0.block(0, 3, 3, 3) * u.transpose();
		sqm.block(3, 0, 3, 3).noalias() = u * mat_sqm0.block(3, 0, 3, 3) * u.transpose();
		sqm.block(3, 3, 3, 3).noalias() = u * mat_sqm0.block(3, 3, 3, 3) * u.transpose();
		efm.block(0, 0, 3, 3).noalias() = u * mat_efm0.block(0, 0, 3, 3) * u.transpose();
		efm.block(3, 0, 3, 3).noalias() = u * mat_efm0.block(3, 0, 3, 3) * u.transpose();
		dipole.noalias() = u * mat_dipole0 * u.transpose();
		this->calculate_efield_batch(E.data(), dE.data(), 1);
		p1.noalias() = dipole * E;
		Ft.block(0, 0, 3, 1).noalias() = (p1.transpose() * dE).transpose();
		Ft.block(3, 0, 3, 1).noalias() = p1.cross(E);
		for (int ri = 0; ri < 6; ri++) {
			randx(ri, 0) = dis(urng);
			//randx(ri, 0) = 0;
		}
		dx.noalias() = co * (sqm * randx);
		accu.addDiffuse(dx);
		dx.block(0, 0, 3, 1).noalias() = pa * (efm.block(0, 0, 3, 3) * E);
		dx.block(3, 0, 3, 1).noalias() = pa * (efm.block(3, 0, 3, 3) * E);
		accu.addShift(dx);
		dx.noalias() = mob * Ft;
		accu.addDipole(dx);
		xyz = accu.applyTo(mat_xyz0);
		u = accu.applyTo(mat_u0);
		

		//for (int j = 0; j < 3; j++) {
		//	utmp.col(j) = u.col(j) + dx.block<3, 1>(3, 0).cross(u.col(j)).eval()
		//		- 1 / 2 * (u.col(j) * (dx.block<3, 1>(3, 0).transpose() * dx.block<3, 1>(3, 0))).eval()
		//		- (u.col(j).transpose() * dx.block<3, 1>(3, 0)) * dx.block<3, 1>(3, 0);
		//}
		//u = utmp;
		//e::JacobiSVD<e::MatrixXd> svd;
		//svd.compute(u, e::ComputeFullV | e::ComputeFullU);
		//u = svd.matrixU() * svd.matrixV().transpose();
	}
}

void Simulator::forward_current(float* dI, double* u9, int step) {
	e::Vector3d E;
	e::Matrix3d dE;
	this->calculate_efield_batch(E.data(), dE.data(), 1);
	e::Map<e::Matrix<double, 3, 3, e::ColMajor>> mat_resist0(resist0);
	e::Vector3d mat_dI;
	for (int i = 0; i < step; i++) {
		e::Map<e::Matrix<double, 3, 3, e::ColMajor>> u(u9 + 9 * i);
		mat_dI = u * mat_resist0 * u.transpose() * E;
		dI[i] = mat_dI(2) / (this->pore_length + 1.6 * this->pore_radius) * 1e21;
	}
};

void Simulator::transition_matrix(double* tpm, double* u9, int step, int skip, int bin_num) {
	e::Vector3d E;
	e::Matrix3d dE;
	this->calculate_efield_batch(E.data(), dE.data(), 1);
	e::Map<e::Matrix<double, 3, 3, e::ColMajor>> mat_resist0(resist0);
	e::Vector3d mat_dI;
	std::vector<double> dI(step);
	double max = -DBL_MAX;
	double min = DBL_MAX;
	for (int i = 0; i < step; i++) {
		e::Map<e::Matrix<double, 3, 3, e::ColMajor>> u(u9 + 9 * i);
		mat_dI = u * mat_resist0 * u.transpose() * E;
		dI[i] = mat_dI(2) / (this->pore_length + 1.6 * this->pore_radius) * 1e21;
		min = (dI[i] < min) ? dI[i] : min;
		max = (dI[i] > max) ? dI[i] : max;
	}

	double bin = (max - min) / double(bin_num);
	memset(tpm, 0, bin_num * bin_num * sizeof(double));
	std::vector<double> tpm_sum(bin_num, 0);
	for (int i = 0; i < step; i++) {
		int idxi = floor((dI[i] - min) / bin);
		idxi = (bin_num <= idxi) ? bin_num - 1 : idxi;
		tpm[bin_num * bin_num + idxi] = (double(idxi) + 0.5) * bin + min;
		tpm[bin_num * bin_num + bin_num + idxi]++;
	}
	for (int i = 0; i < bin_num; i++) {
		tpm[bin_num * bin_num + bin_num + i] /= double(step);
	}
	for (int i = skip; i < step; i += skip) {
		int idxi = floor((dI[i - skip] - min) / bin);
		int idxj = floor((dI[i] - min) / bin);
		idxi = (bin_num <= idxi) ? bin_num - 1 : idxi;
		idxj = (bin_num <= idxj) ? bin_num - 1 : idxj;
		tpm[idxi * bin_num + idxj]++;
		tpm_sum[idxi]++;
	}
	for (int i = 0; i < bin_num; i++) {
		for (int j = 0; j < bin_num; j++) {
			if (tpm_sum[i] != 0)
				tpm[i * bin_num + j] /= tpm_sum[i];
		}
	}
	return;
};

