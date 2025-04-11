#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "solvercuda.cuh"
#include "hydrodynamic_cu.cuh"
#include "electrostatic.cuh"
#include "model.cuh"

void compute_tensor_particle_file(std::string& fn, std::string& fn_out, int Nc, double epp, double epw, int solver_type);

void compute_tensor_shell_file(std::string& fn, std::string& fn_out, int Nc, int num_neigh, double epp, double epw, int solver_type);

void compute_tensor_particle(std::string& fn_xyz, int Nc, double epp, double epw, int solver_type,
	double* mob, double* resist, double* dipole, double* efm);

void compute_tensor_shell(std::string& fn, int Nc, int num_neigh, double epp, double epw, int solver_type, 
	double* mob, double* resist, double* dipole, double* efm);