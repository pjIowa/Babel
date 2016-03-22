#include <iostream>
#include <vector>
#include <string>
#include <armadillo>

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    int numIterations;
        
    public:
    NeuralNetwork(arma::mat i, arma::mat t, int numIt) {
        input = i;
        target = t;
        numIterations = numIt;
    }
};


int main() {
    arma::mat input = {{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};
    arma::mat target = {{0, 1, 1, 0}};
    int numIterations = 1000;
    
    NeuralNetwork model(input, target, numIterations);
    return 0;
}