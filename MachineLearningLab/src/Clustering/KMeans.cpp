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

	std::vector<int> previousLabels(num_samples, -1); 
	std::vector<int> currentLabels(num_samples, -1);

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

	// Randomly assign centroid positions

	std::vector<double> maxValues = (num_features, 0.0);

	for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
		for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
			double value = data[sample_idx][feature_idx];
			double maxValue = maxValues[feature_idx];

			if (value > maxValue) {
				maxValues[feature_idx] = value;
			}
		}
	}

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

	while (iterations < maxIterations_ && clusterChanged) {
		
		for (int centroid_idx = 0; centroid_idx < centroids_.size(); ++centroid_idx) {
			for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
				data[sample_idx];
			}
		}

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