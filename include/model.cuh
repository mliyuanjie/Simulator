#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

class BeadsModel {
public:
	BeadsModel(std::string& fn, int neight_num, int Nc, double epp, double epw, double xyz0, double xyz1, double xyz2);
	BeadsModel(std::vector<double>& xyz, std::vector<double>& rp,
		std::vector<double>& zetap, int Nc, double epp, double epw);
	~BeadsModel() {
		if (d_xyz) cudaFree(d_xyz);
		if (d_rp) cudaFree(d_rp);
		if (d_zetap) cudaFree(d_zetap);
	}
	std::vector<double> xyz;
	std::vector<double> rp;
	std::vector<double> zetap;
	std::vector<double> r_theta_phi;
	double center[3] = {0, 0, 0};
	int Np;
	int Nc;
	double epp;
	double epw;
	double3* d_xyz = nullptr; 
	double* d_rp = nullptr; 
	double* d_zetap = nullptr;
};