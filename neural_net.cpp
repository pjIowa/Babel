#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <armadillo>

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    arma::mat weights;
    int numIterations;

    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }

    arma::mat sigmoid_derivative(arma::mat x) {
        return x * (1-x);
    }

    void randomInitWeights() {
        weights.randu(1, input.n_cols);
        weights.elem( find(weights > 1.0) ).ones();
        weights.elem( find(weights < -1.0) ).fill(-1.0);
    }

    public:
    NeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    }

    void train(int numIt) {
        numIterations = numIt;
        for(int i=0; i<numIterations; i++) {
            arma::mat output = input.each_row() % weights;
        }
    }
};


int main() {
    //4x3
    arma::mat input = {{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};
    //1x4
    arma::mat target = {{0, 1, 1, 0}};
    int numIterations = 1000;

    NeuralNetwork model(input, target);
    model.train(numIterations);
    return 0;
}
