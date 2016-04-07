#include "neuralNet.h"

int main() {
    arma::mat input = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    arma::mat target = {{0, 1, 1, 1, 1, 0, 0}};
    int numIterations = 60000;

    std::cout << "Neural Network trained on XOR examples" << std::endl;
    NeuralNetwork model(input, target);
    model.train(numIterations);
    return 0;
}
