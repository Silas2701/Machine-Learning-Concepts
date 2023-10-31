#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox

Eigen::VectorXd coefficients;

										///  LinearRegression class implementation  ///


// ------ MATRIX COMPUTATION ------

// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/
	
	// TODO
      // Check if the sizes of trainData and trainLabels match
 
    if (trainData.size() != trainLabels.size()) {
        std::cerr << "Error: Size mismatch between trainData and trainLabels." << std::endl;
        return;
    }

    // Convert trainData to matrix representation
    Eigen::MatrixXd X(trainData.size(), trainData[0].size());
    for (int i = 0; i < trainData.size(); ++i) {
        for (int j = 0; j < trainData[0].size(); ++j) {
            X(i, j) = trainData[i][j];
        }
    }

    // Construct the design matrix X (with an additional column of ones for the intercept)
    Eigen::MatrixXd X_with_intercept(trainData.size(), trainData[0].size() + 1);
    X_with_intercept << Eigen::MatrixXd::Ones(trainData.size(), 1), X;

    // Convert trainLabels to matrix representation
    Eigen::VectorXd y(trainLabels.size());
    for (int i = 0; i < trainLabels.size(); ++i) {
        y(i) = trainLabels[i];
    }

    // Calculate the coefficients using the least squares method
     // Store the coefficients for future predictions
    m_coefficients = (X_with_intercept.transpose() * X_with_intercept).ldlt().solve(X_with_intercept.transpose() * y);

}




// ------ GRADIENT DESCENT ------

// Function to fit the linear regression model to the training data using Gradient Descent
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int num_epochs) {
    int num_features = trainData[0].size();
    int num_samples = trainData.size();

    // Initialize weights with zeros
    std::vector<double> weights = std::vector<double>(num_features, 0.0);

    // Iterate over the epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Initialize an empty weight gradients vector holding the current gradient descent change
        std::vector<double> weight_gradients = std::vector<double>(num_features, 0.0);

        // Iterate over all the features one by one
        for (int i = 0; i < num_features; i++) {
            // Iterate over all samples
            for (int j = 0; j < num_samples; j++) {
                // Fetch one sample and the appropriate sample
                std::vector<double> x = trainData[j];
                double y = trainLabels[j];

                // Calculate the predicted value by summing up all the features multiplied by their weight
                double predicted = 0.0;
                for (int k = 0; k < x.size(); k++) {
                    predicted += weights[k] * x[k];
                }

                // Sum up difference between the predicted value and the label and multiply it with the feature value
                weight_gradients[i] += (predicted - y) * x[i];
            }

            // Calculate the new weight by subtracting the gradient difference
            weights[i] -= learning_rate * weight_gradients[i] / num_samples;
        }
    }

    // Create an Eigen Vector for the weights
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(trainData[0].size() + 1);

    for (int i = 0; i < weights.size(); ++i) {
        coefficients(i + 1) = weights[i];
    }

    m_coefficients = coefficients;
}



// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	// TODO

	std::vector<double> result;
   
    // Check if the model has been fitted
    if (m_coefficients.size() == 0) {
        std::cerr << "Error: Model has not been fitted. Please call the 'fit' function first." << std::endl;
        return result;
    }

    // Convert testData to matrix representation
    Eigen::MatrixXd X(testData.size(), testData[0].size());
    for (int i = 0; i < testData.size(); ++i) {
        for (int j = 0; j < testData[0].size(); ++j) {
            X(i, j) = testData[i][j];
        }
    }

    // Construct the design matrix X (with an additional column of ones for the intercept)
    Eigen::MatrixXd X_with_intercept(testData.size(), testData[0].size() + 1);
    X_with_intercept << Eigen::MatrixXd::Ones(testData.size(), 1), X;
	

    // Make predictions using the stored coefficients
    Eigen::VectorXd predictions = X_with_intercept * m_coefficients;


    // Convert predictions to a vector
    for (int i = 0; i < predictions.size(); ++i) {
        result.push_back(predictions(i));
    }

    return result;
}



/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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
        fit(trainData, trainLabels); // Matrice computation
        //fit(trainData, trainLabels,0.000005,100); // Gradient descent 

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