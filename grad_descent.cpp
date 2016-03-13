#include <iostream>
#include <armadillo>
#include <string>
#include <math.h>

double computeLogitError(arma::mat weights, double bias, arma::mat input, arma::mat target);

int main() {
    std::string fileName = "diabetes.csv";
    arma::mat csvData;
    csvData.load(fileName, arma::csv_ascii);
    arma::mat input = csvData.cols(0, 7);
    arma::mat target = csvData.col(8);
    arma::mat weights(8, 1, arma::fill::zeros);
    double bias = 0;
    double logitError = computeLogitError(weights, bias, input, target);
    return 0;
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