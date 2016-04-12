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
    
    //return context for each time step
    arma::mat forwardStep(arma::mat inputWeights, arma::mat contextWeights) {
        arma::mat context = arma::zeros<arma::mat>(input.n_rows, input.n_cols + 1);
        
        for(int i=0; i<input.n_cols; i++) {
            context.col(i+1) = updateContext(input.col(i), context.col(i), inputWeights, contextWeights);
        }
        
        return context;
    }
    
    //take input for time step, previous context, input weights, and context weights
    //return new context
    arma::mat updateContext(arma::mat examplesForTimeStep, arma::mat previousContext, arma::mat inputWeights, arma::mat contextWeights) {
        return examplesForTimeStep*inputWeights + previousContext*contextWeights;
    }
    
    //take context of last time step (prediction)
    //return gradient update for output weights
    arma::mat outputGradient(arma::mat prediction) {
        return 2.0 * (prediction - target) / target.n_rows;
    }
    
    void backwardGradient() {
        //take the input, context for all time steps, output gradient, and context weight
        //go backwards through all time steps
        //return gradient update for input and context weights
    }
    
    void resilientPropagationUpdate() {
        arma::mat inputWeights = weights.col(0);
        arma::mat contextWeights = weights.col(1);
        
        arma::mat context = forwardStep(inputWeights, contextWeights);
        arma::mat gradientOutput = outputGradient(context.col(context.n_cols-1));
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
