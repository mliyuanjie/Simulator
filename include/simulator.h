#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <string>
#include <random>
#include <Eigen/Core> 
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace  e = Eigen;
class Simulator {
public:
	void forward_motion_batch(double* x3, double* u9, int step, int skip, int batch);
	void calculate_force_batch(double** d_dx, double** d_force, double** d_t1,
		const double** d_sqm, const double** d_mob, const double** d_efm, const double** d_dipole,
		const double** d_t1_const, const double** d_force_const, const double** d_E, const double** d_dE, int batch);
	void calculate_efield_batch(double* E, double* dE, int batch);
	void set_position_batch(double* u, double* xyz, int batch);
	void loadfile(std::string& fn_tensor, double dt,
		double temperature, double salt, double salt_resis,
		double pore_radius, double pore_length, double epw);
	void loadtensor(double* mob, double* resist, double* dipole, double* efm, double dt,
		double temperature, double salt, double salt_resis,
		double pore_radius, double pore_length, double epw);
	void forward_motion(double* x3, double* u9, int step, int skip);
	void forward_current(float* dI, double* u9, int step);
	void transition_matrix(double* tpm, double* u9, int step, int skip, int bin_num);

	int batch = 10;
	int step = 100;
	int skip = 1;
	double co = 0;
	double pa = 0;
	double rp = 1e-9;
	double efm0[18];
	double mob0[36];
	double sqm0[36];
	double resist0[9];
	double dipole0[9];
	double xyz0[3] = { 0,0,0 };
	double u0[9] = { 1,0,0,0,1,0,0,0,1 };

	double pore_radius = 0;
	double pore_length = 0;
	

	VSLStreamStatePtr stream;
};