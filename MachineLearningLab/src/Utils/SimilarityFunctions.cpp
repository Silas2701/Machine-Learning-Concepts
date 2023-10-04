#include "SimilarityFunctions.h"
#include <cmath>
#include <stdexcept>


										// SimilarityFunctions class implementation //
			

/// hammingDistance function: Calculates the Hamming distance between two vectors.
double SimilarityFunctions::hammingDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
	if (v1.size() != v2.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	for (std::size_t i = 0; i < v1.size(); ++i) {
		dist += std::abs(v1.at(i) - v2.at(i));
	}

	return dist;
}


/// jaccardDistance function: Calculates the Jaccard distance between two vectors.
double SimilarityFunctions::jaccardDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double num = 0.0;
	double den = 0.0;
	double dist = 0.0;

	// Compute the Jaccard Distance
	// TODO

	// vector to set
	std::set<double> setA(a.begin(), a.end());
	std::set<double> setB(b.begin(), b.end());

	// calculate the intersection
	std::set<double> intersection;
	for (double elem : setA) {
		if (setB.count(elem) > 0) {
			intersection.insert(elem);
		}
	}

	// calculate the union
	std::set<double> unionSet = setA;
	unionSet.insert(setB.begin(), setB.end());

	// Jaccard Distance
	dist = 1.0 - static_cast<double>(intersection.size()) / unionSet.size();

	return dist;
}


/// cosineDistance function: Calculates the cosine distance between two vectors.///
double SimilarityFunctions::cosineDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dotProduct = 0.0;
	double normA = 0.0;
	double normB = 0.0;
	double cosinedist = 0.0;

	// Compute the cosine Distance
	// TODO
	// scalar product
	for (size_t i = 0; i < a.size(); ++i) {
		dotProduct += a[i] * b[i];
	}

	// calcul of the lenght
	for (size_t i = 0; i < a.size(); ++i) {
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	normA = std::sqrt(normA);
	normB = std::sqrt(normB);

	cosinedist = dotProduct / (normA * normB);

	return cosinedist;
}


/// euclideanDistance function: Calculates the Euclidean distance between two vectors.///
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	// Compute the Euclidean Distance
	// TODO
	for (size_t i = 0; i < a.size(); ++i) {
		double difference = a[i] - b[i];
		dist += difference * difference;
	}

	dist = std::sqrt(dist);

	return dist;
}


/// manhattanDistance function: Calculates the Manhattan distance between two vectors.///
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	// Compute the Manhattan Distance
	// TODO

	for (size_t i = 0; i < a.size(); ++i) {
		dist += std::abs(a[i] - b[i]);
	}

	return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.///
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	// Compute the Minkowski Distance
	// TODO
	for (size_t i = 0; i < a.size(); ++i) {
		dist += std::pow(std::abs(a[i] - b[i]), p);
	}

	dist = std::pow(dist, 1.0 / p);

	return dist;
}


