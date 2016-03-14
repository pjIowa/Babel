#include <iostream>
#include <armadillo>
#include <string>
#include <math.h>
#include <utility>

double computeLogitError(arma::mat weights, double bias, arma::mat input, arma::mat target);
std::pair<arma::mat, double> updateParameters(arma::mat weights, double bias, arma::mat input, arma::mat target, double learningRate);
std::vector<double> gradientDescent(arma::mat weights, double bias, arma::mat input, arma::mat target, double initialLearningRate, int numIterations);
void plotData(std::vector<double> data);

int main() {
    int numIterations = 10000;
    std::vector<double> errorList (numIterations, 0.0);
    
    std::string fileName = "diabetes.csv";
    arma::mat csvData;
    csvData.load(fileName, arma::csv_ascii);
    arma::mat input = csvData.cols(0, 7);
    arma::mat target = csvData.col(8);
    arma::mat weights(8, 1, arma::fill::zeros);
    double bias = 0.0;
    double learningRate = 0.001;
    std::cout << "Gradient Descent on " << fileName << std::endl;
    errorList = gradientDescent(weights, bias, input, target, learningRate, numIterations);
    plotData(errorList);
    
    fileName = "myopia.csv";
    csvData.load(fileName, arma::csv_ascii);
    input = csvData.cols(1, 13);
    target = csvData.col(0);
    weights.zeros(13, 1);
    learningRate = 0.0001;
    std::cout << "Gradient Descent on " << fileName << std::endl;
    errorList = gradientDescent(weights, bias, input, target, learningRate, numIterations);
    plotData(errorList);
    
    return 0;
}

void plotData(std::vector<double> data) {
    FILE *pipe = popen("gnuplot -persist" , "w");
    
    if (pipe != NULL) {
        
        fprintf(pipe, "set style line 5 lt rgb 'cyan' lw 3 pt 6 \n");
        fprintf(pipe, "plot '-' with linespoints ls 5 \n");
        
        for (int i=0; i<data.size(); i++) {
            fprintf(pipe, "%lf %lf\n", double(i), data[i]);
        }
        fprintf(pipe, "e");
        
        fflush(pipe);
        pclose(pipe);
    }
    else {
        std::cout << "Could not open gnuplot pipe" << std::endl;
    }
}

std::vector<double> gradientDescent(arma::mat weights, double bias, arma::mat input, arma::mat target, double initialLearningRate, int numIterations) {
    double logitError = computeLogitError(weights, bias, input, target);
    double learningRate = initialLearningRate;
    
    std::vector<double> errorList (numIterations, 0.0);
    
    for(int i=0; i<numIterations; i++) {
        errorList[i] = logitError;
        if (i%1000==0) {
            std::cout << "Step " << i << " " << logitError << std::endl;
        }
        std::pair<arma::mat, double> updatedSet = updateParameters(weights, bias, input, target, learningRate);
        weights = updatedSet.first;
        bias = updatedSet.second;
        logitError = computeLogitError(weights, bias, input, target);
    }
    
    return errorList;
}

double computeLogitError(arma::mat weights, double bias, arma::mat input, arma::mat target) {
    double totalError = 0.0;
    for(int i=0; i<input.n_rows; i++) {
        double weightedSum = bias;
        for(int j=0; j<input.n_cols; j++) {
             weightedSum += weights(j)*input(i,j);
        }
        double prediction = 1.0/(1.0 + exp(-weightedSum));
        totalError += target(i) - prediction;
    }
    return totalError / input.n_rows;
}

std::pair<arma::mat, double> updateParameters(arma::mat weights, double bias, arma::mat input, arma::mat target, double learningRate) {
    double biasGradient = 0.0;
    arma::mat weightGradients(weights.n_rows, 1, arma::fill::zeros);
    double N = input.n_rows;
    for(int i=0; i<N; i++) {
        double weightedSum = bias;
        for(int j=0; j<input.n_cols; j++) {
            weightedSum += weights(j) * input(i,j);
        }
        double prediction = 1.0 / (1.0 + exp(-weightedSum));
        
        biasGradient += -2.0/N * (target[i] - prediction);
        weightGradients += -2.0/N * input[i] * (target[i] - prediction);
    }
    
    arma::mat retWeights = weights - learningRate * weightGradients;
    double retBias = bias - learningRate * biasGradient;
    std::pair<arma::mat, double> retPair(retWeights, retBias);
    return retPair;
}