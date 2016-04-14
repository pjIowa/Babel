#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__
#include <armadillo>

class RecurrentNeuralNetwork {
    int numContextStates = 2;
    arma::mat inputWeights;
    arma::mat contextWeights;
    arma::mat outputWeights;
    
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
        inputWeights.randn(1, numContextStates);
        contextWeights.randn(numContextStates, numContextStates);
        outputWeights.randn(numContextStates, target.n_cols);
    }
    
    //take input for time step, previous context, input weights, and context weights
    //return new context
    arma::mat updateContext(arma::mat examplesForTimeStep, arma::mat previousContext, arma::mat inputWeights, arma::mat contextWeights) {
//        std::cout << size(examplesForTimeStep) << std::endl;
//        std::cout << size(inputWeights) << std::endl;
//        std::cout << size(previousContext) << std::endl;
//        std::cout << size(contextWeights) << std::endl;
//        x*W_xh+ h*W_hh
//        20x1*1x2+20x2*2x2 = 20x2
        return examplesForTimeStep*inputWeights + previousContext*contextWeights;
    }
    
    //take prediction
    //return gradient update for output weights
    arma::mat outputGradient(arma::mat prediction) {
        return 2.0 * (prediction - target) / target.n_rows;
    }
    
    //take context for all time steps, output gradient, and context weight
    //return gradient update for input and context weights
    std::vector<arma::mat> backwardGradient(std::vector<arma::mat> context, arma::mat inputGradientOutput) {
        
        std::vector<arma::mat> iGOT, cGOT;
        iGOT.reserve(input.n_cols+1);
        cGOT.reserve(input.n_cols+1);
        iGOT.back() = gradientOutput;
        
        
//        arma::mat inputGradientOverTime  = arma::zeros<arma::mat>(input.n_rows, input.n_cols + 1);
//        inputGradientOverTime.col(inputGradientOverTime.n_cols-1) = gradientOutput;
        arma::mat inputGradient = { 0 };
        arma::mat contextGradient = arma::zeros<arma::mat>(1, numContextStates);
        
        for(int i=input.n_cols; i>0; i--) {
            std::cout << size(gradientOverTime.col(i)) << std::endl;
            std::cout << size(input.col(i-1)) << std::endl;
            std::cout << size(context[i-1]) << std::endl;
            std::cout << size(contextWeights) << std::endl;
            inputGradient += sum(gradientOverTime.col(i)%input.col(i-1));
            contextGradient += sum(gradientOverTime.col(i)%context[i-1]);
            gradientOverTime.col(i-1) = gradientOverTime.col(i)*contextWeights;
        }
        std::cin.get();
        return std::vector<arma::mat> { inputGradient, contextGradient };
    }
    
    double resilientPropagationUpdate() {
        //Forward calculations
        std::vector<arma::mat> context = forwardStep();
//        std::cout << size(context.back()) << std::endl;
//        std::cout << size(outputWeights) << std::endl;
//        std::cout << size(target) << std::endl;
        arma::mat gradientOutput = outputGradient(context.back()*outputWeights);
//        std::cout << gradientOutput << std::endl;
        
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
    std::vector<arma::mat> forwardStep() {
        std::vector<arma::mat> ctx;
        ctx.push_back(arma::zeros<arma::mat>(input.n_rows, numContextStates));
        for(int i=0; i<input.n_cols; i++) {
            ctx.push_back(updateContext(input.col(i), ctx[i], inputWeights, contextWeights));
        }
        return ctx;
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
