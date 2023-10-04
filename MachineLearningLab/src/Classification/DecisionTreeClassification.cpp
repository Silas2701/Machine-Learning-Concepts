#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	
	//traverseTree()
	
	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	// Loop through candidate features and potential split thresholds
    double parent_entropy = EntropyFunctions::entropy(y);

    int num_samples = X.size();
    int num_features = X[0].size();
    double best_gain = -1.0;
    int split_idx = -1; 
    double split_thresh = 0.0;

	if (num_samples == 1) {
		return new Node(-1, split_thresh, nullptr, nullptr, y[0]);
	}

	for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
		for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
			double candidate_threshold = X[sample_idx][feature_idx];

			double information_gain = informationGain(y, X[feature_idx], candidate_threshold);

			// Check if this is the best split so far
			if (information_gain > best_gain) {
				best_gain = information_gain;
				split_idx = feature_idx;
				split_thresh = candidate_threshold;
			}
		}
	}

	X.erase(X.begin() + split_idx);

	// Split the data into left and right based on the best split
	std::vector<std::vector<double>> X_left;
	std::vector<std::vector<double>> X_right;

	std::vector<double> y_left;
	std::vector<double> y_right;

	for (int i = 0; i < num_samples; i++) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}
	// Recursively grow the left and right subtrees
	Node* left = growTree(X_left, y_left, depth + 1);
	Node* right = growTree(X_right, y_right, depth + 1);

	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);

	// Initialize variables to keep track of child entropies and weights
	std::vector<int> left_idxs; 
	std::vector<int> right_idxs;
	int left_child_count = 0;
	int right_child_count = 0;

	// Iterate through the data points and calculate entropies for left and right children
	for (size_t i = 0; i < y.size(); i++) {
		if (X_column[i] <= split_thresh) {
			left_idxs.push_back(i);
			left_child_count++;
		}
		else {
			right_idxs.push_back(i);
			right_child_count++;
		}
	}

	// Calculate the weighted average of child entropies
	double left_child_entropy = EntropyFunctions::entropy(X_column, left_idxs);
	double right_child_entropy = EntropyFunctions::entropy(X_column, right_idxs);

	// Calculate information gain as the difference between parent entropy and weighted child entropies
	double ig = parent_entropy - ((static_cast<double>(left_child_count) / y.size()) * left_child_entropy
		+ (static_cast<double>(right_child_count) / y.size()) * right_child_entropy);

	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;

	std::unordered_map<double, int> label_map;
	
	for (const double& label : y) {
		if (label_map.find(label) == label_map.end()) {
			label_map[label] = 1;
		}
		else {
			label_map[label]++;
		}
	}

	auto max_pair = std::max_element(
		std::begin(label_map), 
		std::end(label_map), 
		[](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
			return p1.second < p2.second;
		}
	);

	most_common = max_pair->second;

	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {
	// If the node is a leaf node, return its value
	if (node->isLeafNode()) {
		return node->value;
	}

	// Get the feature index associated with this node
	int feature_index = node->feature;

	// Check if the feature value of the input vector is less than or equal to the node's threshold
	if (x[feature_index] <= node->threshold) {
		// Traverse the left subtree
		return traverseTree(x, node->left);
	}
	else {
		// Traverse the right subtree
		return traverseTree(x, node->right);
	}
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}