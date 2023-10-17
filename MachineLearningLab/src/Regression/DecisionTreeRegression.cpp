#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{

}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));

	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	
	// Implement the function
	
	for (std::vector<double>& x : X) {
		double prediction = traverseTree(x, root);

		predictions.push_back(prediction);
	}

	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	int num_samples = X.size();
	int num_features = X[0].size();
	double best_gain = std::numeric_limits<double>::infinity();
	int split_idx = -1;
	double split_thresh = 0.0;

	// Restructure the original Matrix X to easily access all the features in one column
	std::vector<std::vector<double>> X_columns;

	for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
		std::vector<double> X_column;

		for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
			X_column.push_back(X[sample_idx][feature_idx]);
		}

		X_columns.push_back(X_column);
	}

	// Check every value in the feature matrix for the best split value
	for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
		for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
			double candidate_threshold = X[sample_idx][feature_idx];

			// Find appropriate split point by minimizing mean squared error 
			double mean_squared_error = meanSquaredError(y, X_columns[feature_idx], candidate_threshold);

			// Check if this is the best split so far
			if (mean_squared_error < best_gain) {
				best_gain = mean_squared_error;
				split_idx = feature_idx;
				split_thresh = candidate_threshold;
			}
		}
	}

	// Stopping criteria
	double error = meanSquaredError(y, X_columns[split_idx], split_thresh);
	double errorThreshold = 10;

	if (error < errorThreshold) {
		return new Node(0, 0.0, nullptr, nullptr, mean(y));
	}

	// Split the data into left and right based on the best split
	std::vector<std::vector<double>> X_left;
	std::vector<std::vector<double>> X_right;

	std::vector<double> y_left;
	std::vector<double> y_right;

	for (int i = 0; i < num_samples; i++) {
		std::vector<double> sample = X[i];
		double label = y[i];

		double split_value = X[i][split_idx];

		// Ignore values that are infinity for example after removing them from the matrix
		sample[split_idx] = std::numeric_limits<double>::infinity();

		// If split value is smaller than the split threshold/split point put it into the left node otherwise to the right
		if (split_value <= split_thresh) {
			X_left.push_back(sample);
			y_left.push_back(label);
		}
		else {
			X_right.push_back(sample);
			y_right.push_back(label);
		}
	}

	Node* left;
	Node* right;

	// If the maximum depth is reached or the sample size is not enough, don't split and take the mean value
	if (depth == max_depth - 1 || y_left.size() < min_samples_split) {
		left = new Node(0, 0.0, nullptr, nullptr, mean(y_left));
	}
	else {
		left = growTree(X_left, y_left, depth + 1);
	}

	// If the maximum depth is reached or the sample size is not enough, don't split and take the mean value
	if (depth == max_depth - 1 || y_right.size() < min_samples_split) {
		right = new Node(0, 0.0, nullptr, nullptr, mean(y_right));
	}
	else {
		right = growTree(X_right, y_right, depth + 1);
	}

	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double mse = 0.0;

	double totalNumber = static_cast<double>(X_column.size());

	std::vector<double> y_left;
	std::vector<double> y_right;

	for (int i = 0; i < X_column.size(); ++i) {
		double label = y[i];

		if (X_column[i] <= split_thresh) {
			y_left.push_back(label);
		}
		else {
			y_right.push_back(label);
		}
	}

	double meanLeft = mean(y_left);
	double countLeft = static_cast<double>(y_left.size());
	double squaredLeft = 0.0;

	for (const int y : y_left) {
		squaredLeft += std::pow(y - meanLeft, 2);
	}

	double mseLeft = 0.0;

	if (countLeft != 0)
		mseLeft = (countLeft / totalNumber) * squaredLeft;

	double meanRight = mean(y_right);
	double countRight = static_cast<double>(y_right.size());
	double squaredRight = 0.0;

	for (const int y : y_right) {
		squaredRight += std::pow(y - meanRight, 2);
	}

	double mseRight = 0.0;

	if (countRight != 0)
		mseRight = (countRight / totalNumber) * squaredRight;

	mse = mseLeft + mseRight;
	
	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;

	double totalValue = 0.0;
	for (const int value : values) {
		totalValue += value;
	}

	meanValue = totalValue / static_cast<double>(values.size());
	
	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {
	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/

	if (node->isLeafNode())
		return node->value;

	int feature_index = node->feature;
	double threshold = node->threshold;

	if (x[feature_index] <= threshold)
		return traverseTree(x, node->left);
	else
		return traverseTree(x, node->right);
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

