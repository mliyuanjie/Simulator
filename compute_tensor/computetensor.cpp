#include "computetensor.h" 
#include "mkl.h"
#include "mkl_lapacke.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

constexpr double pi = 3.1415926535897932384626433;

std::vector<std::complex<double>> compute_dipole_moment_F(int Np, int nharm, std::vector<std::complex<double>>& g) {
	int dim = Np * nharm;
	std::complex<double> xj(0.0, 1.0);
	std::vector<std::complex<double>> pdip(9, 0);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < Np; i++) {
			pdip[2 * 3 + j] += g[i * nharm + 1 + j * dim] * pi * 4.0;
			pdip[j] += (g[i * nharm + j * dim] / sqrt(2.0) - g[i * nharm + 2 + j * dim] * sqrt(2.0)) * 2.0 * sqrt(2.0) * pi;
			pdip[1 * 3 + j] += (g[i * nharm + j * dim] / sqrt(2.0) + g[i * nharm + 2 + j * dim] * sqrt(2.0)) * 2.0 * sqrt(2.0) * pi / xj;
		}
	}
	return pdip;
}

std::vector<std::complex<double>> compute_electrical_resistance_F(int Np, int nharm, std::vector<double> &rp, std::vector<std::complex<double>>& Bi) {
	int dim = Np * nharm;
	std::complex<double> xj(0.0, 1.0);
	std::vector<std::complex<double>> Erm(9, 0);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < Np; i++) {
			Erm[j] -= 2.0 / 3.0 * pi * pow(rp[i], 3) * Bi[i * nharm + j * dim];
			Erm[j] += 4.0 / 3.0 * pi * pow(rp[i], 3) * Bi[i * nharm + 2 + j * dim];
			Erm[3 + j] += 2.0 / 3.0 * xj * pi * pow(rp[i], 3) * Bi[i * nharm + j * dim];
			Erm[3 + j] += 4.0 / 3.0 * xj * pi * pow(rp[i], 3) * Bi[i * nharm + 2 + j * dim];
			Erm[6 + j] -= 4.0 / 3.0 * pi * pow(rp[i], 3) * Bi[i * nharm + 1 + j * dim];
		}
	}
	return Erm;
}

std::vector<std::complex<double>> compute_hydrodynamic_resistance_F(int Np, int nharm, double* center, std::vector<double>& xyz, std::vector<std::complex<double>>& yb) {
	int dim = Np * nharm;
	double cnlm_1_1 = 2.8944050182330705;
	double cnlm_1_0 = 2.0466534158929770;
	std::complex<double> xj(0.0, 1.0);
	std::vector<std::complex<double>> Re(36, 0);
	for (int j = 0; j < 6; j++) {
		for (int i = 0; i < Np; i++) {
			std::complex<double> Fx = -3.0 / 2.0 * cnlm_1_1 * (yb[i * nharm + 2 + j * dim] - yb[i * nharm + 8 + j * dim]);
			std::complex<double> Fy = 3.0 * xj / 2.0 * cnlm_1_1 * (yb[i * nharm + 2 + j * dim] + yb[i * nharm + 8 + j * dim]);
			std::complex<double> Fz = -3.0 / 2.0 * cnlm_1_0 * 2.0 * yb[i * nharm + 5 + j * dim];
			Re[j] -= Fx;
			Re[6 + j] -= Fy;
			Re[12 + j] -= Fz;
			std::complex<double> Tx = -3.0 / 2.0 * cnlm_1_1 * (yb[i * nharm + 1 + j * dim] - yb[i * nharm + 7 + j * dim]);
			std::complex<double> Ty = 3.0 * xj / 2.0 * cnlm_1_1 * (yb[i * nharm + 1 + j * dim] + yb[i * nharm + 7 + j * dim]);
			std::complex<double> Tz = -3.0 / 2.0 * cnlm_1_0 * 2.0 * yb[i * nharm + 4 + j * dim];
			double posx = xyz[i * 3] - center[0];
			double posy = xyz[i * 3 + 1] - center[1];
			double posz = xyz[i * 3 + 2] - center[2];
			std::vector<double> force = {Fx.real(), Fy.real(), Fz.real()};
			std::vector<double> torque = crossp(posx, posy, posz, force);
			Re[18 + j] -= (Tx + torque[0]);
			Re[24 + j] -= (Ty + torque[1]);
			Re[30 + j] -= (Tz + torque[2]);
		}
	}
	for (int i = 0; i < 36; i++) {
		Re[i] /= 6.0 * pi;
	}

	std::vector<std::complex<double>> rid(36, 0.0);
	for (int i = 0; i < 6; i++) {
		rid[i * 6 + i] = std::complex<double>(1.0, 0.0);
	}
	std::vector<int> ipiv(6);
	LAPACKE_zgetrf(LAPACK_ROW_MAJOR, 6, 6, reinterpret_cast<MKL_Complex16*>(Re.data()), 6, ipiv.data());
	LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', 6, 6, reinterpret_cast<MKL_Complex16*>(Re.data()), 6, ipiv.data(), reinterpret_cast<MKL_Complex16*>(rid.data()), 6);

	return rid;
}

std::vector<std::complex<double>> compute_electrophoretic_mobility_F(int Np, int nharm, double* center, std::vector<double>& xyz, std::vector<std::complex<double>> rid, std::vector<std::complex<double>>& yb) {
	int dim = Np * nharm;
	double cnlm_1_1 = 2.8944050182330705;
	double cnlm_1_0 = 2.0466534158929770;
	std::complex<double> xj(0.0, 1.0);
	std::vector<std::complex<double>> Ref(18, 0);
	for (int j = 6; j < 9; j++) {
		for (int i = 0; i < Np; i++) {
			std::complex<double> Fx = -3.0 / 2.0 * cnlm_1_1 * (yb[i * nharm + 2 + j * dim] - yb[i * nharm + 8 + j * dim]);
			std::complex<double> Fy = 3.0 * xj / 2.0 * cnlm_1_1 * (yb[i * nharm + 2 + j * dim] + yb[i * nharm + 8 + j * dim]);
			std::complex<double> Fz = -3.0 / 2.0 * cnlm_1_0 * 2.0 * yb[i * nharm + 5 + j * dim];
			Ref[j - 6] -= Fx;
			Ref[3 + j - 6] -= Fy;
			Ref[6 + j - 6] -= Fz;
			std::complex<double> Tx = -3.0 / 2.0 * cnlm_1_1 * (yb[i * nharm + 1 + j * dim] - yb[i * nharm + 7 + j * dim]);
			std::complex<double> Ty = 3.0 * xj / 2.0 * cnlm_1_1 * (yb[i * nharm + 1 + j * dim] + yb[i * nharm + 7 + j * dim]);
			std::complex<double> Tz = -3.0 / 2.0 * cnlm_1_0 * 2.0 * yb[i * nharm + 4 + j * dim];
			double posx = xyz[i * 3] - center[0];
			double posy = xyz[i * 3 + 1] - center[1];
			double posz = xyz[i * 3 + 2] - center[2];
			std::vector<double> force = { Fx.real(), Fy.real(), Fz.real() };
			std::vector<double> torque = crossp(posx, posy, posz, force);
			Ref[9 + j - 6] -= (Tx + torque[0]);
			Ref[12 + j - 6] -= (Ty + torque[1]);
			Ref[15 + j - 6] -= (Tz + torque[2]);
		}
	}
	for (int i = 0; i < 18; i++) {
		Ref[i] /= -6.0 * pi;
	}
	std::vector<std::complex<double>> Elm(18, 0.0);
	MKL_Complex16 alpha = { 1.0, 0.0 };
	MKL_Complex16 beta = { 0.0, 0.0 };
	cblas_zgemm(CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		6,
		3,
		6,
		&alpha,
		rid.data(), 6,
		Ref.data(), 3,
		&beta,
		Elm.data(), 3);
	return Elm;
}

void compute_tensor_particle_file(std::string& fn, std::string& fn_out, int Nc, double epp, double epw, int solver_type) {
	std::string line;
	std::vector<int> index;
	std::vector<double> xyzrc;
	std::ifstream file(fn);
	while (std::getline(file, line)) {
		std::istringstream item(line);
		double num;
		item >> num;
		if (int(num) == 0)
			continue;
		index.push_back(num);
		for (int i = 0; i < 5; i++) {
			item >> num;
			xyzrc.push_back(num);
		}
	}
	int Np = index.size();
	std::vector<double> xyz;
	std::vector<double> rp;
	std::vector<double> zetap;
	for (int i = 0; i < Np; i++) {
		if (index[i] == 0)
			continue;
		xyz.push_back(xyzrc[i * 5]);
		xyz.push_back(xyzrc[i * 5 + 1]);
		xyz.push_back(xyzrc[i * 5 + 2]);
		rp.push_back(xyzrc[i * 5 + 3]);
		zetap.push_back(xyzrc[i * 5 + 4]);
	}
	BeadsModel model(xyz, rp, zetap, Nc, epp, epw);

	ElectroStatic elec(model);
	Hydrodynamic hydro(model);
	if (model.Np < 100) {
		solver_type = 0;
	}
	Solver solve_elec(elec, solver_type);
	solve_elec.solve("electrical resistance");
	std::vector<std::complex<double>> g;
	std::vector<std::complex<double>> Be;
	std::vector<std::complex<double>> Bi;
	std::vector<std::complex<double>> yb;
	solve_elec.get_x(g);
	solve_elec.get_y(Be);
	solve_elec.matmult(1.0, 1.0);
	solve_elec.get_y(Bi);
	//dipole moment 
	std::vector<std::complex<double>> Pdip = compute_dipole_moment_F(elec.Np, elec.nharm, g);
	//electrical resistance
	std::vector<std::complex<double>> Erm = compute_electrical_resistance_F(elec.Np, elec.nharm, model.rp, Bi);
	solve_elec.clear();

	Solver solve_hydro(hydro, solver_type);
	hydro.set_slip_condition(Be, g);
	solve_hydro.solve("hydrodynamics");
	solve_hydro.get_x(yb);
	//hydrodynamic resistance matrix and electrophoretic
	std::vector<std::complex<double>> Rid = compute_hydrodynamic_resistance_F(hydro.Np, hydro.nharm, model.center, model.xyz, yb);
	std::vector<std::complex<double>> Elm = compute_electrophoretic_mobility_F(hydro.Np, hydro.nharm, model.center, model.xyz, Rid, yb);
	//write down
	{
		std::ofstream ofs(fn_out, std::ios::out | std::ios::trunc);
		std::string line;
		ofs << std::scientific<< std::setprecision(13);
		ofs << "\n# Result Electric_Resistance_matrix\n";
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Erm[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Mobility_Matrix_Rigid_Body\n";
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				ofs << Rid[i * 6 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Electrophoretic_Matrix_Rigid_Body\n";
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Elm[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Dipole_moment_matrix\n";
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Pdip[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs.close();
	}
}

void compute_tensor_particle(std::string& fn_xyz, int Nc, double epp, double epw, int solver_type, 
	double* mob, double* resist, double* dipole, double* efm) {
	std::vector<int> index;
	std::vector<double> xyzrc;
	std::ifstream file(fn_xyz);
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream item(line);
		double num;
		item >> num;
		if (int(num) == 0)
			continue;
		index.push_back(num);
		for (int i = 0; i < 5; i++) {
			item >> num;
			xyzrc.push_back(num);
		}
	}
	int Np = index.size();
	std::vector<double> xyz;
	std::vector<double> rp;
	std::vector<double> zetap;
	for (int i = 0; i < Np; i++) {
		xyz.push_back(xyzrc[i * 5]);
		xyz.push_back(xyzrc[i * 5 + 1]);
		xyz.push_back(xyzrc[i * 5 + 2]);
		rp.push_back(xyzrc[i * 5 + 3]);
		zetap.push_back(xyzrc[i * 5 + 4]);
	}
	BeadsModel model(xyz, rp, zetap, Nc, epp, epw);
	ElectroStatic elec(model);
	Hydrodynamic hydro(model);
	if (model.Np < 100) {
		solver_type = 0;
	}
	Solver solve_elec(elec, solver_type);
	solve_elec.solve("electrical resistance");
	std::vector<std::complex<double>> g;
	std::vector<std::complex<double>> Be;
	std::vector<std::complex<double>> Bi;
	std::vector<std::complex<double>> yb;
	solve_elec.get_x(g);
	solve_elec.get_y(Be);
	solve_elec.matmult(1.0, 1.0);
	solve_elec.get_y(Bi);
	//dipole moment 
	std::vector<std::complex<double>> Pdip = compute_dipole_moment_F(elec.Np, elec.nharm, g);
	//electrical resistance
	std::vector<std::complex<double>> Erm = compute_electrical_resistance_F(elec.Np, elec.nharm, model.rp, Bi);
	solve_elec.clear();

	Solver solve_hydro(hydro, solver_type);
	hydro.set_slip_condition(Be, g);
	solve_hydro.solve("hydrodynamics");
	solve_hydro.get_x(yb);
	//hydrodynamic resistance matrix and electrophoretic
	std::vector<std::complex<double>> Rid = compute_hydrodynamic_resistance_F(hydro.Np, hydro.nharm, model.center, model.xyz, yb);
	std::vector<std::complex<double>> Elm = compute_electrophoretic_mobility_F(hydro.Np, hydro.nharm, model.center, model.xyz, Rid, yb);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			resist[j * 3 + i] = Erm[i * 3 + j].real();
			dipole[j * 3 + i] = Pdip[i * 3 + j].real();
		}
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 6; ++j) {
			mob[j * 6 + i] = Rid[i * 6 + j].real();
		}
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 3; ++j) {
			efm[j * 6 + i] = Elm[i * 3 + j].real();
		}
	}
}

void compute_tensor_shell(std::string& fn_xyz, int Nc, int num_neigh, 
	double epp, double epw, int solver_type, double* mob, double* resist, double* dipole, double* efm) {
	std::vector<int> index;
	std::vector<double> xyzrc;
	std::ifstream file(fn_xyz);
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream item(line);
		double num;
		item >> num;
		if (int(num) == 0)
			continue;
		index.push_back(num);
		for (int i = 0; i < 5; i++) {
			item >> num;
			xyzrc.push_back(num);
		}
	}
	int Np = index.size();
	std::vector<int> barcode;
	std::vector<double> xyz_shell;
	std::vector<double> xyz_particle;
	std::vector<double> rp_shell;
	std::vector<double> rp_particle;
	std::vector<double> zetap_shell;
	std::vector<double> zetap_particle;
	for (int i = 0; i < Np; i++) {
		xyz_particle.push_back(xyzrc[i * 5]);
		xyz_particle.push_back(xyzrc[i * 5 + 1]);
		xyz_particle.push_back(xyzrc[i * 5 + 2]);
		rp_particle.push_back(xyzrc[i * 5 + 3]);
		zetap_particle.push_back(xyzrc[i * 5 + 4]);
		barcode.push_back(0);
		if (index[i] < num_neigh) {
			xyz_shell.push_back(xyzrc[i * 5]);
			xyz_shell.push_back(xyzrc[i * 5 + 1]);
			xyz_shell.push_back(xyzrc[i * 5 + 2]);
			rp_shell.push_back(xyzrc[i * 5 + 3]);
			zetap_shell.push_back(xyzrc[i * 5 + 4]);
			barcode[barcode.size() - 1] = 1;
		}
	}
	BeadsModel model_shell(xyz_shell, rp_shell, zetap_shell, Nc, epp, epw);
	BeadsModel model_particle(xyz_particle, rp_particle, zetap_particle, Nc, epp, epw);
	ElectroStatic elec(model_particle);
	Hydrodynamic hydro(model_shell);
	if (model_particle.Np < 100) {
		solver_type = 0;
	}
	Solver solve_elec(elec, solver_type);
	solve_elec.solve("electrical resistance");
	std::vector<std::complex<double>> g;
	std::vector<std::complex<double>> Be;
	std::vector<std::complex<double>> Bi;
	std::vector<std::complex<double>> yb;
	solve_elec.get_x(g);
	solve_elec.get_y(Be);
	solve_elec.matmult(1.0, 1.0);
	solve_elec.get_y(Bi);
	//dipole moment 
	std::vector<std::complex<double>> Pdip = compute_dipole_moment_F(elec.Np, elec.nharm, g);
	//electrical resistance
	std::vector<std::complex<double>> Erm = compute_electrical_resistance_F(elec.Np, elec.nharm, model_particle.rp, Bi);
	solve_elec.clear();

	Solver solve_hydro(hydro, solver_type);
	hydro.set_slip_condition(Be, g);
	solve_hydro.solve("hydrodynamics");
	solve_hydro.get_x(yb);
	//hydrodynamic resistance matrix and electrophoretic
	std::vector<std::complex<double>> Rid = compute_hydrodynamic_resistance_F(hydro.Np, hydro.nharm, model_shell.center, model_shell.xyz, yb);
	std::vector<std::complex<double>> Elm = compute_electrophoretic_mobility_F(hydro.Np, hydro.nharm, model_shell.center, model_shell.xyz, Rid, yb);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			resist[j * 3 + i] = Erm[i * 3 + j].real();
			dipole[j * 3 + i] = Pdip[i * 3 + j].real();
		}
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 6; ++j) {
			mob[j * 6 + i] = Rid[i * 6 + j].real();
		}
	}
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 3; ++j) {
			efm[j * 6 + i] = Elm[i * 3 + j].real();
		}
	}
}

void compute_tensor_shell_file(std::string& fn, std::string& fn_out, int Nc, int num_neigh, 
	double epp, double epw, int solver_type) {
	std::string line;
	std::vector<int> index;
	std::vector<double> xyzrc;
	std::ifstream file(fn);
	while (std::getline(file, line)) {
		std::istringstream item(line);
		double num;
		item >> num;
		if (int(num) == 0)
			continue;
		index.push_back(num);
		for (int i = 0; i < 5; i++) {
			item >> num;
			xyzrc.push_back(num);
		}
	}
	int Np = index.size();
	std::vector<int> barcode;
	std::vector<double> xyz_shell;
	std::vector<double> xyz_particle;
	std::vector<double> rp_shell;
	std::vector<double> rp_particle;
	std::vector<double> zetap_shell;
	std::vector<double> zetap_particle;
	for (int i = 0; i < Np; i++) {
		xyz_particle.push_back(xyzrc[i * 5]);
		xyz_particle.push_back(xyzrc[i * 5 + 1]);
		xyz_particle.push_back(xyzrc[i * 5 + 2]);
		rp_particle.push_back(xyzrc[i * 5 + 3]);
		zetap_particle.push_back(xyzrc[i * 5 + 4]);
		barcode.push_back(0);
		if (index[i] < num_neigh) {
			xyz_shell.push_back(xyzrc[i * 5]);
			xyz_shell.push_back(xyzrc[i * 5 + 1]);
			xyz_shell.push_back(xyzrc[i * 5 + 2]);
			rp_shell.push_back(xyzrc[i * 5 + 3]);
			zetap_shell.push_back(xyzrc[i * 5 + 4]);
			barcode[barcode.size() - 1] = 1;
		}
	}
	BeadsModel model_shell(xyz_shell, rp_shell, zetap_shell, Nc, epp, epw);
	BeadsModel model_particle(xyz_particle, rp_particle, zetap_particle, Nc, epp, epw);
	ElectroStatic elec(model_particle);
	Hydrodynamic hydro(model_shell);
	if (model_particle.Np < 100) {
		solver_type = 0;
	}
	Solver solve_elec(elec, solver_type);
	solve_elec.solve("electrical resistance");
	std::vector<std::complex<double>> g;
	std::vector<std::complex<double>> Be;
	std::vector<std::complex<double>> Bi;
	std::vector<std::complex<double>> yb;
	solve_elec.get_x(g);
	solve_elec.get_y(Be);
	solve_elec.matmult(1.0, 1.0);
	solve_elec.get_y(Bi);
	//dipole moment 
	std::vector<std::complex<double>> Pdip = compute_dipole_moment_F(elec.Np, elec.nharm, g);
	//electrical resistance
	std::vector<std::complex<double>> Erm = compute_electrical_resistance_F(elec.Np, elec.nharm, model_particle.rp, Bi);
	solve_elec.clear();

	Solver solve_hydro(hydro, solver_type);
	hydro.set_slip_condition(Be, g);
	solve_hydro.solve("hydrodynamics");
	solve_hydro.get_x(yb);
	//hydrodynamic resistance matrix and electrophoretic
	std::vector<std::complex<double>> Rid = compute_hydrodynamic_resistance_F(hydro.Np, hydro.nharm, model_shell.center, model_shell.xyz, yb);
	std::vector<std::complex<double>> Elm = compute_electrophoretic_mobility_F(hydro.Np, hydro.nharm, model_shell.center, model_shell.xyz, Rid, yb);
	//write down
	{
		std::ofstream ofs(fn_out, std::ios::out | std::ios::trunc);
		std::string line;
		ofs << std::scientific << std::setprecision(13);
		ofs << "\n# Result Electric_Resistance_matrix\n";
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Erm[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Mobility_Matrix_Rigid_Body\n";
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				ofs << Rid[i * 6 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Electrophoretic_Matrix_Rigid_Body\n";
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Elm[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs << "# Result Dipole_moment_matrix\n";
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				ofs << Pdip[i * 3 + j].real() << " ";
			}
			ofs << "\n";
		}
		ofs.close();
	}
}