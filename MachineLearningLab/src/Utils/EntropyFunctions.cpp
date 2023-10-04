#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;
	
	for (const double& label : y) {
		if (label_map.find(label) == label_map.end()) {
			label_map[label] = 1;
		}
		else {
			label_map[label]++;
		}
	}
	
	for (const auto& pair : label_map) {
		double probability = static_cast<double>(pair.second) / total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;

	// Convert labels to unique integers and count their occurrences
	for (int i : idxs) {
		double label = y[i];
		if (label_map.find(label) == label_map.end()) {
			label_map[label] = 1;
		}
		else {
			label_map[label]++;
		}
	}

	// Compute the probability and entropy
	for (const auto& pair : label_map) {
		double probability = static_cast<double>(pair.second) / total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}


