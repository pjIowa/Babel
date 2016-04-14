#include "rnn.h"

int main (void) {
    int sequenceLength = 10;
    int numberOfSequences = 20;
    arma::mat input = round(arma::randu<arma::mat>(numberOfSequences, sequenceLength));
    arma::mat target =  sum(input, 1);
    
    std::cout << std::endl;
//    std::cout << "Recurrent Neural Network trained to sum an input sequence" << std::endl;
    RecurrentNeuralNetwork model(input, target);
    std::clock_t startTime = std::clock();
    std::vector<double> costVector = model.train(1700);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << std::endl;
    
    for(int i=0; i<costVector.size(); i+=200) {
        std::cout << "Step " << i << "  \t" << costVector[i] << std::endl;
    }
    std::cout << std::endl;
    
    
    arma::mat testInput = {{0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1}};
    model.input = testInput;
    arma::mat testOutput = model.forwardStep();
    std::cout << std::endl;
    std::cout << "Expectation on Test Sequence: " << accu(testInput) << std::endl;
    std::cout << "Prediction:                   " << as_scalar(testOutput.col(testOutput.n_cols-1)) << std::endl;
    std::cout << std::endl;
    return 0;
}
