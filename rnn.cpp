#include "rnn.h"

int main (void) {
    int sequenceLength = 10;
    int numberOfSequences = 20;
    arma::mat input = round(arma::randu<arma::mat>(numberOfSequences, sequenceLength));
    arma::mat target =  sum(input, 1);
    
    RecurrentNeuralNetwork model(input, target);
    std::clock_t startTime = std::clock();
    model.train(500);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    
    //TODO: add test data
    return 0;
}
