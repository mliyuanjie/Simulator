/*
compute the mobility tensor in rigids cluster beads simulation in nanopore 
Sparse Matrix, CPU OR GPU;
1. compute the hydrodynamic mobility tensor based on the fortran code.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include "simulator.h"
#include "computetensor.h"
#include "transitionmatrix.h"

void simulatesinglefile_particle(std::string& fn, int Nc, double epp, double epw, int solver_type, double dt,
    double salt, double temperature, double salt_resist, double pore_length, double pore_radius, int step) {
    std::string fn_out = fn.substr(0, fn.size() - 4) + "_tensor.txt";
    compute_tensor_particle_file(fn, fn_out, Nc, epp, epw, solver_type);
    Simulator sim;
    sim.loadfile(fn_out, dt, temperature, salt, salt_resist, pore_radius, pore_length, epw);
    std::vector<double> x(3 * step);
    std::vector<float> dI(step);
    std::vector<double> u(9 * step);
    auto start = std::chrono::high_resolution_clock::now();
    sim.forward_motion(x.data(), u.data(), step, 1);
    sim.forward_current(dI.data(), u.data(), step);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    std::string fn2 = fn.substr(0, fn.size() - 4) + "_xyz.dat";
    std::string fn3 = fn.substr(0, fn.size() - 4) + "_u.dat";
    std::string fn4 = fn.substr(0, fn.size() - 4) + "_current.dat";
    std::ofstream ofs(fn2, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs2(fn3, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs3(fn4, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(x.data()), 3 * step * sizeof(double));
    ofs2.write(reinterpret_cast<const char*>(u.data()), 9 * step * sizeof(double));
    ofs3.write(reinterpret_cast<const char*>(dI.data()), step * sizeof(float));
}

void simulatesinglefile_shell(std::string& fn, int Nc, int num_neigh, double epp, double epw, int solver_type, double dt,
    double salt, double temperature, double salt_resist, double pore_length, double pore_radius, int step) {
    std::string fn_out = fn.substr(0, fn.size() - 4) + "_tensor.txt";
    compute_tensor_shell_file(fn, fn_out, Nc, num_neigh, epp, epw, solver_type);
    Simulator sim;
    sim.loadfile(fn_out, dt, temperature, salt, salt_resist, pore_radius, pore_length, epw);
    std::vector<double> x(3 * step);
    std::vector<float> dI(step);
    std::vector<double> u(9 * step);
    auto start = std::chrono::high_resolution_clock::now();
    sim.forward_motion(x.data(), u.data(), step, 1);
    sim.forward_current(dI.data(), u.data(), step);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    std::string fn2 = fn.substr(0, fn.size() - 4) + "_xyz.dat";
    std::string fn3 = fn.substr(0, fn.size() - 4) + "_u.dat";
    std::string fn4 = fn.substr(0, fn.size() - 4) + "_current.dat";
    std::ofstream ofs(fn2, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs2(fn3, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs3(fn4, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(x.data()), 3 * step * sizeof(double));
    ofs2.write(reinterpret_cast<const char*>(u.data()), 9 * step * sizeof(double));
    ofs3.write(reinterpret_cast<const char*>(dI.data()), step * sizeof(float));
}

void runtensor_particle(std::string& fn, int Nc, double epp, double epw, int solver_type) {
    std::string fn_out = fn.substr(0, fn.size() - 4) + "_tensor.txt";
    compute_tensor_particle_file(fn, fn_out, Nc, epp, epw, solver_type);
}

void runtensor_shell(std::string& fn, int Nc, int num_neigh, double epp, double epw, int solver_type) {
    std::string fn_out = fn.substr(0, fn.size() - 4) + "_tensor.txt";
    compute_tensor_shell_file(fn, fn_out, Nc, num_neigh, epp, epw, solver_type);
}

void bd_simulate(std::string& fn, double epw, double dt,
    double salt, double temperature, double salt_resist, double pore_length, double pore_radius, int step) {
    Simulator sim;
    sim.loadfile(fn, dt, temperature, salt, salt_resist, pore_radius, pore_length, epw);
    std::vector<double> x(3 * step);
    std::vector<float> dI(step);
    std::vector<double> u(9 * step);
    auto start = std::chrono::high_resolution_clock::now();
    sim.forward_motion(x.data(), u.data(), step, 1);
    sim.forward_current(dI.data(), u.data(), step);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    std::string fn2 = fn.substr(0, fn.size() - 4) + "_xyz.dat";
    std::string fn3 = fn.substr(0, fn.size() - 4) + "_u.dat";
    std::string fn4 = fn.substr(0, fn.size() - 4) + "_current.dat";
    std::ofstream ofs(fn2, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs2(fn3, std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream ofs3(fn4, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(x.data()), 3 * step * sizeof(double));
    ofs2.write(reinterpret_cast<const char*>(u.data()), 9 * step * sizeof(double));
    ofs3.write(reinterpret_cast<const char*>(dI.data()), step * sizeof(float));
}

void simulatesingle_particle(std::string& fn, int Nc, double epp, double epw, int solver_type, double dt,
    double salt, double temperature, double salt_resist, double pore_length, double pore_radius, int step) {
    double mob[36];
    double resist[9];
    double dipole[9];
    double efm[18];
    compute_tensor_particle(fn, Nc, epp, epw, solver_type, mob, resist, dipole, efm);
    Simulator sim;
    sim.loadtensor(mob, resist, dipole, efm, dt, temperature, salt, salt_resist, pore_radius, pore_length, epw);
    std::vector<double> x(3 * step);
    std::vector<float> dI(step);
    std::vector<double> u(9 * step);
    auto start = std::chrono::high_resolution_clock::now();
    sim.forward_motion(x.data(), u.data(), step, 1);
    sim.forward_current(dI.data(), u.data(), step);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    std::string fn4 = fn.substr(0, fn.size() - 4) + "_current.dat";
    std::ofstream ofs3(fn4, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs3.write(reinterpret_cast<const char*>(dI.data()), step * sizeof(float));
}

void simulatesingle_shell(std::string& fn, int Nc, int num_neigh, double epp, double epw, int solver_type, double dt,
    double salt, double temperature, double salt_resist, double pore_length, double pore_radius, int step) {
    double mob[36];
    double resist[9];
    double dipole[9];
    double efm[18];
    compute_tensor_shell(fn, Nc, num_neigh, epp, epw, solver_type, mob, resist, dipole, efm);
    Simulator sim;
    sim.loadtensor(mob, resist, dipole, efm, dt, temperature, salt, salt_resist, pore_radius, pore_length, epw);
    std::vector<double> x(3 * step);
    std::vector<float> dI(step);
    std::vector<double> u(9 * step);
    auto start = std::chrono::high_resolution_clock::now();
    sim.forward_motion(x.data(), u.data(), step, 1);
    sim.forward_current(dI.data(), u.data(), step);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    std::string fn4 = fn.substr(0, fn.size() - 4) + "_current.dat";
    std::ofstream ofs3(fn4, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs3.write(reinterpret_cast<const char*>(dI.data()), step * sizeof(float));
}

int main(int argc, char* argv[]) {
    /*
	//std::string fn = "./parameters.txt";
	//if (argc >= 2) {
	//	fn = argv[1];
	//}
    std::string fn[10] = {
        "./result/current_0.dat",
        "./result/current_1.dat",
        "./result/current_2.dat",
        "./result/current_3.dat",
        "./result/current_4.dat",
        "./result/current_5.dat",
        "./result/current_6.dat",
        "./result/current_7.dat",
        "./result/current_8.dat",
        "./result/current_9.dat",
    };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        std::cout << i << ": ";
        std::vector<float> cur(100000);
        std::ifstream filei(fn[i], std::ios::binary);
        filei.read(reinterpret_cast<char*>(cur.data()), 100000 * sizeof(float));
        std::vector<float> cura(50000);
        std::vector<float> curb(50000);
        memcpy(cura.data(), cur.data(), 50000 * sizeof(float));
        for (int j = 0; j < 10; j++) {
            std::vector<float> cur2(100000);
            std::ifstream filej(fn[j], std::ios::binary);
            filej.read(reinterpret_cast<char*>(cur2.data()), 100000 * sizeof(float));
            memcpy(curb.data(), cur2.data() + 50000, 50000 * sizeof(float));
            double prob = compute_transition_matrix(cura, 50.0, 2, curb, 0.05);
            //double prob = compute_mean_std(cura, 50.0, curb);
            std::cout << std::fixed << std::setprecision(3) << prob << " ";
        }
        
        //for (int j = 0; j < 10; j++) {
        //    std::vector<float> curb(100000);
         //   std::ifstream filej(fn[j], std::ios::binary);
         //   filej.read(reinterpret_cast<char*>(curb.data()), 100000 * sizeof(float));
         //   double prob = compute_transition_matrix(cura, 50.0, 1, curb, 0.05);
            //double prob = compute_mean_std(cura, 50.0, curb);
         //   std::cout << std::fixed << std::setprecision(3) << prob << " ";
        //}
        std::cout << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "|---" << "BD time" << "---|---" << duration.count() / 1000.0 << " s---|" << std::endl;
    */
    
    
    //if (argc >= 2) {
    //	fn = argv[1];
    //}

    std::string fn = "./input.txt";
    if (argc >= 2) {
        fn = argv[1];
    }
    int Nc = 4;
    double epp = 2;
    double epw = 78.5;
    int solver_type = 2;
    double dt = 500.0;
    double salt = 2.0;
    double temperature = 20.0;
    double salt_resist = 0.046;
    double pore_radius = 10.0;
    double pore_length = 30.0;
    int step = 100000000;
    int num_neigh = 6;
    simulatesinglefile_shell(fn, Nc, num_neigh, epp, epw, solver_type, dt,
        salt, temperature, salt_resist, pore_length, pore_radius, step);
    return 0;
}