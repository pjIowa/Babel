#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__
#include <armadillo>

class RecurrentNeuralNetwork {
    arma::mat weights;
    arma::mat weightDeltas;
    arma::mat previousWeightSigns;
    
    double etaP = 1.2;
    double etaN = 0.5;
    
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
        weights.randn(1, 2);
    }
    
    void forwardStep() {
        //take input, inputweight, and contextweight
        //combine and return context for each time step
    }
    
    void outputGradient() {
        //take context of last time step and target
        //return gradient update for the output weights
    }
    
    void backwardGradient() {
        //take the input, context for all time steps, output gradient, and context weight
        //go backwards through all time steps
        //return gradient update for input and context weights
    }
    
    void resilientPropagationUpdate() {
        forwardStep();
        outputGradient();
        backwardGradient();
        //get sign of weight gradients
        for(int i=0; i<weights.n_cols; i++) {
            //if new weight sign is same as previous weight sign
                //multiply delta by etaP
            //else
                //multiply delta by etaN
        }
        //store new signs in previous sign
        for(int i=0; i<weights.n_cols; i++) {
            arma::mat weight = weights.col(i);
            //subtract sign*delta from weight
        }
    }
    
    public:
    arma::mat input;
    arma::mat target;
    
    RecurrentNeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
        weightDeltas = {0.001, 0.001};
        previousWeightSigns = {0, 0};
    } 
    
    void train(int numIt) {
        
        resilientPropagationUpdate();
        
//        for(int i=0; i<numIt; i++) {
//            
//        }
        
//        
//        for(int t=0; t<input.n_rows; t++) {
//        }
    }
};
#endif
