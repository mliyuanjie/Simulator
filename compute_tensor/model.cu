#include "model.cuh" 

BeadsModel::BeadsModel(std::string& fn, int neight_num, int Nc, double epp, double epw, double xyz0, double xyz1, double xyz2) {
	Np = 0;
	//read xyz
	std::ifstream fin(fn);
	std::string line;
	std::getline(fin, line);
	std::istringstream item(line);
	this->Nc = Nc;
	this->epp = epp;
	this->epw = epw;
	Np = 0;
	center[0] = xyz0;
	center[1] = xyz1;
	center[2] = xyz2;
	while (std::getline(fin, line)) {
		std::istringstream item(line);
		double num;
		if (item >> num && num <= neight_num) {
			for (int i = 0; i < 3; i++) {
				if (item >> num)
					xyz.push_back(num);
			}
			if (item >> num)
				rp.push_back(num);
			if (item >> num)
				zetap.push_back(num);
			Np++;
		}	
	}

	r_theta_phi = std::vector<double>(3 * Np * Np);
	for (int i = 0; i < Np; i++) {
		for (int j = 0; j < Np; j++) {
			double dx = xyz[i * 3] - xyz[j * 3];
			double dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
			double dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];
			r_theta_phi[i * 3 * Np + j * 3] = sqrt(dx * dx + dy * dy + dz * dz);
			r_theta_phi[i * 3 * Np + j * 3 + 1] = dz / r_theta_phi[i * 3 * Np + j * 3];
			r_theta_phi[i * 3 * Np + j * 3 + 2] = atan2(dy, dx);
		}
	}
	cudaMalloc(&d_xyz, sizeof(double3) * Np * Np);
	cudaMemcpy((void*)d_xyz, (void*)r_theta_phi.data(), sizeof(double3) * Np * Np, cudaMemcpyHostToDevice);
	//radius of particle
	cudaMalloc(&d_rp, sizeof(double) * Np);
	cudaMemcpy((void*)d_rp, (void*)rp.data(), sizeof(double) * Np, cudaMemcpyHostToDevice);
	cudaMalloc(&d_zetap, sizeof(double) * Np);
	cudaMemcpy((void*)d_zetap, (void*)zetap.data(), sizeof(double) * Np, cudaMemcpyHostToDevice);
}

BeadsModel::BeadsModel(std::vector<double>& xyz, std::vector<double>& rp,
	std::vector<double>& zetap, int Nc, double epp, double epw) {
	Np = rp.size();
	this->Nc = Nc;
	this->epp = epp;
	this->epw = epw;

	this->xyz.resize(xyz.size());
	memcpy(this->xyz.data(), xyz.data(), xyz.size() * sizeof(double));
	this->rp.resize(rp.size());
	memcpy(this->rp.data(), rp.data(), rp.size() * sizeof(double));
	this->zetap.resize(zetap.size());
	memcpy(this->zetap.data(), zetap.data(), zetap.size() * sizeof(double));
	r_theta_phi = std::vector<double>(3 * Np * Np);
	for (int i = 0; i < Np; i++) {
		for (int j = 0; j < Np; j++) {
			double dx = xyz[i * 3] - xyz[j * 3];
			double dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
			double dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];
			r_theta_phi[i * 3 * Np + j * 3] = sqrt(dx * dx + dy * dy + dz * dz);
			r_theta_phi[i * 3 * Np + j * 3 + 1] = dz / r_theta_phi[i * 3 * Np + j * 3];
			r_theta_phi[i * 3 * Np + j * 3 + 2] = atan2(dy, dx);
		}
	}

	cudaMalloc(&d_xyz, sizeof(double3) * Np * Np);
	cudaMemcpy((void*)d_xyz, (void*)r_theta_phi.data(), sizeof(double3) * Np * Np, cudaMemcpyHostToDevice);
	cudaMalloc(&d_rp, sizeof(double) * Np);
	cudaMemcpy((void*)d_rp, (void*)rp.data(), sizeof(double) * Np, cudaMemcpyHostToDevice);
	cudaMalloc(&d_zetap, sizeof(double) * Np);
	cudaMemcpy((void*)d_zetap, (void*)zetap.data(), sizeof(double) * Np, cudaMemcpyHostToDevice);
}