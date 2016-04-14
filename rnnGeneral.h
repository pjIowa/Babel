#ifndef __RNNGENERAL_H__
#define __RNNGENERAL_H__
#include <armadillo>

class RecurrentNeuralNetwork {
    
    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }
    
    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }
    
    arma::mat activate(arma::mat a, arma::mat b) {
        return sigmoid(a*b);
    }
    
    void randomInitWeights() {
        arma::arma_rng::set_seed(1);
    }
    
    double propagationUpdate() {
        double error = 0.0;
        
        return error;
    }
    
    public:
    arma::mat input;
    arma::mat target;
    
    RecurrentNeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    } 
    
    std::vector<arma::mat> forwardStep() {
        std::vector<arma::mat> ctx;
        return ctx;
    }
    
    std::vector<double> train(int numIt) {
        std::vector<double> costVector;
        for(int i=0; i<numIt; i++) {
            double cost = propagationUpdate();
            costVector.push_back(cost);
        }
        return costVector;
    }
};
#endif
