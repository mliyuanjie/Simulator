#include "solvercuda.cuh"
#include <cusolverDn.h>
#include <chrono>
#include <iomanip>
#include <fstream>

__global__ void checkSymmetry(const cuDoubleComplex* A, int n, bool* isSymmetric) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && j < n && i < j) {
		double x1 = A[i * n + j].x - A[j * n + i].x;
		double x2 = A[i * n + j].y - A[j * n + i].y;
		if ((abs(x1) + abs(x2)) > 1e-8) {
			*isSymmetric = false;
		}
	}
}

void Solver::Schur() {
	size_t allsize = sizeA + sizeV * 2 + sizeS + lu_work_size * sizeof(cuDoubleComplex);
	allsize /= 1024 * 1024;
	std::cout << "|---Schur---|---GPU---" << std::fixed << std::setprecision(2) << 
		double(allsize) / 1024.0 << " GB---|" << std::endl;
	cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
	cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
	cudaMemset(d_A, 0, sizeA);
	cudaMemset(d_B, 0, sizeS);
	cudaMemset(d_y, 0, sizeV);
	cudaMemset(d_x, 0, sizeV);

	matrix.compute_matrix(d_A, 0, NpA, 0, NpA, false);
	matrix.compute_matrix(d_B, 0, NpA, NpA, Np, false);
	matrix.right_side_F(d_y, 0, NpA);
	matrix.right_side_F(d_y + offsetS, NpA, Np);

	// [A  B]  [x1]  ->  [b1]
	// [C  D]  [x2]  ->  [b2]
	//LU->A
	cusolverDnZgetrf(cusolverH, dimA, dimA, d_A, dimA, lu_work, devIpiv, devInfo);
	//b1 = A'b1
	cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, dimA, cols, d_A, dimA, devIpiv, d_y, dimA, devInfo);
	//B = A'B
	cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, dimA, dimS, d_A, dimA, devIpiv, d_B, dimA, devInfo);
	//D -> A
	matrix.compute_matrix(d_A, NpA, Np, NpA, Np, false);
	matrix.compute_matrix(lu_work, NpA, Np, 0, NpA, false);
	//A = A - C * B
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimS, dimS, dimA,
		&alpha, lu_work, dimS, d_B, dimA, &beta, d_A, dimS);
	//b2 = b2 - C * b1
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimS, cols, dimA,
		&alpha, lu_work, dimS, d_y, dimA, &beta, d_y + offsetS, dimS);
	//LU->S
	cusolverDnZgetrf(cusolverH, dimS, dimS, d_A, dimS, lu_work, devIpiv, devInfo);
	//solve x2
	cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, dimS, cols, d_A, dimS, devIpiv, d_y + offsetS, dimS, devInfo);
	//x1 = b1 - B * x2
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimA, cols, dimS,
		&alpha, d_B, dimA, d_y + offsetS, dimS, &beta, d_y, dimA);
	cudaMemcpy2D(d_x, dim * sizeof(cuDoubleComplex),
		d_y, dimA * sizeof(cuDoubleComplex), dimA * sizeof(cuDoubleComplex), cols,
		cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(d_x + dimA, dim * sizeof(cuDoubleComplex),
		d_y + offsetS, dimS * sizeof(cuDoubleComplex), dimS * sizeof(cuDoubleComplex), cols,
		cudaMemcpyDeviceToDevice);
	matrix.right_side_F(d_y, 0, Np);
}

void Solver::LU() {
	std::cout << "|---LU---|---GPU---" << 
		std::fixed << std::setprecision(2) << 
		double(sizeA + sizeV * 2) / 1024.0 / 1024.0 / 1024.0 + 16.0 * 
		double(lu_work_size) / 1024.0 / 1024.0 / 1024.0 << " GB---|" << std::endl;
	matrix.compute_matrix(d_A, 0, Np, 0, Np, false);
	
	matrix.right_side_F(d_x, 0, Np);
	
	cusolverDnZgetrf(cusolverH, dim, dim, d_A, dim, lu_work, devIpiv, devInfo);

	int h_devInfo = 0;
	cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (h_devInfo != 0) {
		std::cerr << "|---Solving linear system failed " << h_devInfo << "---|" << std::endl;
	}
	cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, dim, cols, d_A, dim, devIpiv, d_x, dim, devInfo);
	h_devInfo = 0;
	cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (h_devInfo != 0) {
		std::cerr << "|---Solving linear system failed " << h_devInfo << "---|" << std::endl;
	}
	matrix.right_side_F(d_y, 0, Np);
}

void Solver::LU_mkl() {
	std::cout << "|---LU---|---CPU---" << std::fixed << std::setprecision(2) << 
		double(dim * dim * 16 + sizeV) / 1024.0 / 1024.0 / 1024.0 << " GB---|" << std::endl;
	size_t dpitch = dim * sizeof(cuDoubleComplex);
	for (int i = 0; i < Np; i += 100) {
		int si = i;
		int ei = (i + 100 <= Np) ? i + 100 : Np;
		size_t spitch = (ei - si) * nharm * sizeof(cuDoubleComplex);
		for (int j = 0; j < Np; j += 100) {
			int sj = j;
			int ej = (j + 100 <= Np) ? j + 100 : Np;
			matrix.compute_matrix(d_A, si, ei, sj, ej, false);
			size_t offset = dim * j * nharm + i * nharm;
			cudaError_t err = cudaMemcpy2D((void*)(c_A + offset), dpitch, d_A, spitch, spitch, (ej - sj) * nharm, cudaMemcpyDeviceToHost);
		}
	}
	matrix.right_side_F_mkl(c_x, 0, Np);
	std::vector<int> ipiv(dim);
	LAPACKE_zgetrf(LAPACK_COL_MAJOR, dim, dim, c_A, dim, ipiv.data());
	LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', dim, cols, c_A, dim, ipiv.data(), c_x, dim);
	cudaMemcpy(d_x, c_x, sizeV, cudaMemcpyHostToDevice);
	matrix.right_side_F(d_y, 0, Np);
}

void Solver::verify(std::string& fn){
	cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
	cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
	matrix.right_side_F(d_y, 0, Np);
	matrix.matmult(d_x, -1, d_y, 1);
	double normmc = 1.0;
	cublasDznrm2(handle, dim * cols, d_y, 1, &normmc);
	std::cout << "| verify | norm2: " << normmc << " |" << std::endl;
	if (fn == "")
		return;
	cuDoubleComplex* M_t;
	std::vector<std::complex<double>> M(dim * dim);
	cudaMalloc(&M_t, dim * dim * sizeof(cuDoubleComplex));
	matrix.compute_matrix(M_t, 0, Np, 0, Np, false);
	cudaMemcpy(M.data(), M_t, dim * dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	//compare_matrix_F(fn, dim, M, 1e-8);
	cudaFree(M_t);
}

void Solver::solve(const char* str) {
	auto start = std::chrono::high_resolution_clock::now();
	if (solver_type == 0)
		LU();
	else if (solver_type == 2)
		Schur();
	else if (solver_type == 1)
		LU_mkl();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "|---" << str <<"---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
	//cudaMemcpy(x_c.data(), d_x, dim * cols * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

void Solver::matmult(double alpha, double beta) {
	matrix.matmult(d_x, alpha, d_y, beta);
}

void Solver::clear() {
	if (d_A) {
		cudaFree(d_A); d_A = nullptr;
	}
	if (d_B) {
		cudaFree(d_B); d_B = nullptr;
	}
	if (d_x) {
		cudaFree(d_x); d_x = nullptr;
	}
	if (d_y) {
		cudaFree(d_y); d_y = nullptr;
	}
	if (lu_work) {
		cudaFree(lu_work); lu_work = nullptr;
	}
	if (devIpiv) {
		cudaFree(devIpiv); devIpiv = nullptr;
	}
	if (c_A) {
		delete[] c_A;
		c_A = nullptr;
	}
	if (c_x) {
		delete[] c_x;
		c_x = nullptr;
	}
}

void Solver::initial() {
	if (solver_type == 2) {
		cudaMalloc(&d_A, sizeA);
		cudaMalloc(&d_B, sizeS);
		cudaMalloc(&d_y, sizeV);
		cudaMalloc(&d_x, sizeV);
		cudaMalloc(&lu_work, lu_work_size * sizeof(cuDoubleComplex));
		cudaMalloc(&devIpiv, dimA * sizeof(int));
	}
	else if (solver_type == 0) {
		cudaMalloc(&d_A, sizeA);
		cudaMalloc(&d_y, sizeV);
		cudaMalloc(&d_x, sizeV);
		cusolverDnZgetrf_bufferSize(cusolverH, dim, dim, d_A, dim, &lu_work_size);
		cudaMalloc(&lu_work, lu_work_size * sizeof(cuDoubleComplex));
		cudaMalloc(&devIpiv, dim * sizeof(int));
	}
	else if (solver_type == 1) {
		cudaMalloc(&d_A, sizeA);
		cudaMalloc(&d_y, sizeV);
		cudaMalloc(&d_x, sizeV);
		c_A = new MKL_Complex16[dim * dim];
		c_x = new MKL_Complex16[dim * cols];
	}
}
