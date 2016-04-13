#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__
#include <armadillo>

class RecurrentNeuralNetwork {
    int numContextStates = 1;
    arma::mat inputWeights;
    arma::mat contextWeights;
    
    arma::mat inputWeightDelta;
    arma::mat contextWeightDeltas;
    
    arma::mat previousInputWeightSigns;
    arma::mat previousContextWeightSigns;
    
    double etaP = 1.2;
    double etaN = 0.5;
    
//    arma::mat sigmoid_derivative(arma::mat x) {
//        return x % (1-x);
//    }
//    
//    arma::mat sigmoid(arma::mat x) {
//        return 1.0 / (1.0 + arma::exp(-1.0*x));
//    }
//    
//    arma::mat activate(arma::mat a, arma::mat b) {
//        return sigmoid(a*b);
//    }
    
    void randomInitWeights() {
        arma::arma_rng::set_seed(1);
        inputWeights.randn(1, 1);
        contextWeights.randn(1, numContextStates);
    }
    
    //take input for time step, previous context, input weights, and context weights
    //return new context
    arma::mat updateContext(arma::mat examplesForTimeStep, arma::mat previousContext, arma::mat inputWeights, arma::mat contextWeights) {
//        std::cout << size(examplesForTimeStep) << std::endl;
//        std::cout << size(inputWeights) << std::endl;
//        std::cout << size(previousContext) << std::endl;
//        std::cout << size(contextWeights) << std::endl;
//        20x1 
//        1x1
//        20x1
//        1x2
//        W_hh*h + W_xh*x
        //20x20*20x2 + 20x1*1x2 = 20x2 
        //2x20*20x20 + 2x1*1x20 = 2x20 
        
        //20x1*1x1 + 20x1*1x2 
        return examplesForTimeStep*inputWeights + previousContext*contextWeights;
    }
    
    //take context of last time step (prediction)
    //return gradient update for output weights
    arma::mat outputGradient(arma::mat prediction) {
        return 2.0 * (prediction - target) / target.n_rows;
    }
    
    //take context for all time steps, output gradient, and context weight
    //return gradient update for input and context weights
    std::vector<arma::mat> backwardGradient(arma::mat context, arma::mat gradientOutput) {
        arma::mat gradientOverTime = arma::zeros<arma::mat>(input.n_rows, input.n_cols + 1);
        gradientOverTime.col(gradientOverTime.n_cols-1) = gradientOutput;
        arma::mat inputGradient = { 0 };
        arma::mat contextGradient = arma::zeros<arma::mat>(1, numContextStates);
        
        for(int i=input.n_cols; i>0; i--) {
            inputGradient += sum(gradientOverTime.col(i)%input.col(i-1));
            contextGradient += sum(gradientOverTime.col(i)%context.col(i-1));
            gradientOverTime.col(i-1) = gradientOverTime.col(i)*contextWeights;
        }
        return std::vector<arma::mat> { inputGradient, contextGradient };
    }
    
    double resilientPropagationUpdate() {
        //Forward calculations
        arma::mat context = forwardStep();
        arma::mat gradientOutput = outputGradient(context.col(context.n_cols-1));
        
        //Backward calculations
        std::vector<arma::mat> weightGradients = backwardGradient(context, gradientOutput);
        
        //Get the sign (-1, 0, 1) of the weight updates
        std::vector<arma::mat> weightGradientSigns;
        for(int i=0; i<weightGradients.size(); i++) {
            weightGradientSigns.push_back(sign(weightGradients[i]));
        }
        arma::mat inputGradientSigns = weightGradientSigns[0];
        arma::mat contextGradientSigns = weightGradientSigns[1];
        
        //Update magnitude of weight changes
        inputWeightDelta.elem( find(inputGradientSigns == previousInputWeightSigns) )*etaP;
        inputWeightDelta.elem( find(inputGradientSigns != previousInputWeightSigns) )*etaN;
        contextWeightDeltas.elem( find(contextGradientSigns == previousContextWeightSigns) )*etaP;
        contextWeightDeltas.elem( find(contextGradientSigns != previousContextWeightSigns) )*etaN;
        previousInputWeightSigns = inputGradientSigns;
        previousContextWeightSigns = contextGradientSigns;
        
        //Update weights
        inputWeights -= inputGradientSigns%inputWeightDelta;
        contextWeights -= contextGradientSigns%contextWeightDeltas;
        
        return accu(gradientOutput);
    }
    
    public:
    arma::mat input;
    arma::mat target;
    
    RecurrentNeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
        
        //TODO: make sizes dynamic
        inputWeightDelta = {0.001};
        contextWeightDeltas.set_size(1, numContextStates);
        contextWeightDeltas.fill(0.001);
        previousInputWeightSigns = {0};
        previousContextWeightSigns.zeros(1, numContextStates);
    } 
    
    //return context for each time step
    arma::mat forwardStep() {
        arma::mat context = arma::zeros<arma::mat>(input.n_rows, input.n_cols + 1);
        for(int i=0; i<input.n_cols; i++) {
            context.col(i+1) = updateContext(input.col(i), context.col(i), inputWeights, contextWeights);
        }
        return context;
    }
    
    std::vector<double> train(int numIt) {
        std::vector<double> costVector;
        for(int i=0; i<numIt; i++) {
            double cost = resilientPropagationUpdate();
            costVector.push_back(cost);
        }
        return costVector;
    }
};
#endif
