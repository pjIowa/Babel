#include "rnn.h"

int main (void) {
    arma::mat input =  {0, 0, 1, 1, 0};
    arma::mat target =  {0, 0, 1, 0, 1};
    
    RecurrentNeuralNetwork model(input, target);
    model.train(5000);
    return 0;
}
