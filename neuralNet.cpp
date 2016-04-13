#include "neuralNet.h"

int main() {
    arma::mat input = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    arma::mat target = {{0, 1, 1, 1, 1, 0, 0}};
    int numIterations = 60000;

    std::cout << std::endl;
    std::cout << "Neural Network trained on XOR examples" << std::endl;
    NeuralNetwork model(input, target);
    std::clock_t startTime = std::clock();
    std::vector<double> costVector = model.train(numIterations);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << std::endl;
    for(int i=0; i<costVector.size(); i+=6000) {
        std::cout << "Step " << i << "  \t" << costVector[i] << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
