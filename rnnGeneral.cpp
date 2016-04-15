#include "rnnGeneral.h"

double isXOR(double a, double b);

double isXOR(double a, double b) {
    double notA = a==0;
    double notB = b==0;
    return (a==notB) | (notA==b);
}

int main (void) {
    int sequenceLength = 10;
    int numberOfSequences = 20;
    
    arma::mat input = round(arma::randu<arma::mat>(numberOfSequences, sequenceLength));
    arma::mat target = arma::zeros<arma::mat>(numberOfSequences, sequenceLength-1);
    for(int i=0; i<input.n_rows; i++) {
        for(int j=0; j<input.n_cols-1; j++) { 
            target(i, j) = isXOR( input(i, j), input(i, j+1) ); 
        }
    }
    
    std::cout << std::endl;
    std::cout << "Non-Linear Recurrent Neural Network trained to XOR a sequence" << std::endl;
    RecurrentNeuralNetwork model(input, target);
    std::clock_t startTime = std::clock();
    std::vector<double> costVector = model.train(1700);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << std::endl;
    
    for(int i=0; i<costVector.size(); i+=200) {
        std::cout << "Step " << i << "  \t" << costVector[i] << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
