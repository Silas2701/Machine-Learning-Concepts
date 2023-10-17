#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {
	srand((unsigned)time(NULL));
}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	//std::vector<std::vector<double>> normalizedData = data;
	int num_samples = data.size();
	int num_features = data[0].size();

	std::map<int, std::vector<int>> previousLabels; 
	std::map<int, std::vector<int>> currentLabels;

	/* Implement the following:
		---	Initialize centroids randomly
		--- Randomly select unique centroid indices
		---	Perform K-means clustering
		--- Assign data points to the nearest centroid
		--- Calculate the Euclidean distance between the point and the current centroid
		--- Update newCentroids and clusterCounts
		--- Update centroids
		---  Check for convergence
	*/

	// For every feature find the max value
	std::vector<double> maxValues(num_features, 0.0);

	for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
		for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
			double value = data[sample_idx][feature_idx];
			double maxValue = maxValues[feature_idx];

			if (value > maxValue) {
				maxValues[feature_idx] = value;
			}
		}
	}

	// Create centroids/seed points to start from by picking a random value for every feature of the centroid 
	// which lies between 0 and the max possible value of the feature for the training data
	for (int i = 0; i < numClusters_; ++i) {
		std::vector<double> centroid;

		for (int j = 0; j < num_features; ++j) {
			double randValue = fmod(rand(), maxValues[j]);

			centroid.push_back(randValue);
		}

		centroids_.push_back(centroid);
	}

	// Start with checking for the closests centroid for each data point and adjust position of the centroid

	int iterations = 0;
	bool clusterChanged = true;

	// Stop criteria: Iterate n times or cluster did not change anymore to the previous one
	while (iterations < maxIterations_ && clusterChanged) {
		// After having compared all labels from the current iteration with the labels from the previous iteration,
		// copy all labels from currentLabels to previousLabels because we start a new iteration
		// and we want to compare the labels from the upcoming iteration with the labels from the previous iteration then
		previousLabels.clear();
		previousLabels.insert(currentLabels.begin(), currentLabels.end());

		// Clear current labels as it will be filled again by new data
		currentLabels.clear();

		// Iterate over all the samples
		for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
			std::vector<double> sample = data[sample_idx];

			int closestCentroid = -1;
			double closestDistance = std::numeric_limits<double>::infinity();

			// Iterate over all centroids and check for all the centroids which one is the closest
			for (int centroid_idx = 0; centroid_idx < centroids_.size(); ++centroid_idx) {
				std::vector<double> centroid = centroids_[centroid_idx];

				// Calculate euclidean distance between the sample and the centroid
				double distance = SimilarityFunctions::euclideanDistance(sample, centroid);

				// Update the cluster id to which the sample belongs to by checking which cluster is the closest
				if (distance < closestDistance) {
					closestDistance = distance;
					closestCentroid = centroid_idx;
				}
			}

			if (closestCentroid != -1) {
				auto it = currentLabels.find(closestCentroid);

				// Safe index of the sample based by their closests centroids in a map
				if (it == currentLabels.end()) {
					currentLabels[closestCentroid] = std::vector<int> { sample_idx };
				} else {
					currentLabels[closestCentroid].push_back(sample_idx);
				}
			}
		}

		// Recalculate the clusters center by estimating new cluster centers
		for (const auto& label : currentLabels) {
			int centroid_idx = label.first;
			std::vector<int> samplesClosestToCentroid = label.second;
			int sampleCount = samplesClosestToCentroid.size();

			std::vector<double> centroid(num_features, 0.0);

			for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
				for (const int sample_idx : samplesClosestToCentroid) {
					centroid[feature_idx] += data[sample_idx][feature_idx];
				}

				centroid[feature_idx] /= static_cast<double>(sampleCount);
			}

			centroids_[centroid_idx] = centroid;
		}

		// Increment the count of iterations by one and compare the previous label set with the current label set
		++iterations;
		clusterChanged = (previousLabels != currentLabels);
	}
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());

	/* Implement the following:
		--- Initialize the closest centroid and minimum distance to the maximum possible value
		--- Iterate through each centroid
		--- Calculate the Euclidean distance between the point and the centroid
		--- Add the closest centroid to the labels vector
	*/
	
	for (auto instance : data) {
		double best_distance = std::numeric_limits<double>::infinity();
		int centroid_idx = -1;

		for (int i = 0; i < centroids_.size(); ++i) {
			double distance = SimilarityFunctions::euclideanDistance(instance, centroids_[i]);

			if (distance < best_distance) {
				best_distance = distance;
				centroid_idx = i;
			}
		}

		labels.push_back(centroid_idx);
	}
	
	return labels; // Return the labels vector
}





/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);

		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}