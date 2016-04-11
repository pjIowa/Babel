#include "rnn.h"

int main (void) {
    int sequenceLength = 10;
    int numberOfSequences = 20;
    arma::mat input = round(arma::randu<arma::mat>(numberOfSequences, sequenceLength));
    arma::mat target =  sum(input, 1);
    
    RecurrentNeuralNetwork model(input, target);
    model.train(5000);
    return 0;
}
