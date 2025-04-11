#pragma once
#include <vector>

double compute_transition_matrix(std::vector<float>& cura, double sigma, int step, 
	std::vector<float>& curb, double curmin);

double compute_mean_std(std::vector<float>& cura, double sigma, std::vector<float>& curb);
