#ifndef __RNNGENERAL_H__
#define __RNNGENERAL_H__
#include <armadillo>

class RecurrentNeuralNetwork {
    
    double learningRate = 0.1;
    int numContextNodes = 2;
    arma::mat inputWeights, contextWeights, outputWeights;
    
    //Neural network boilerplate
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
        inputWeights.randn(1, numContextNodes);
        contextWeights.randn(numContextNodes, numContextNodes);
        outputWeights.randn(numContextNodes, target.n_cols);
    }
    
    //Forward step methods
    arma::mat updateContext(arma::mat inputRow, arma::mat contextRow) {
        return inputRow*inputWeights + contextRow*contextWeights;
    }
    
    std::vector<arma::mat> contextOverTime() {
        std::vector<arma::mat> cOT;
        cOT.push_back(arma::zeros<arma::mat>(input.n_rows, numContextNodes));
        for(int i=0; i<input.n_cols; i++) {
            cOT.push_back(updateContext(input.col(i), cOT[i]));
        }
        return cOT;
    }
    
    double cost(arma::mat prediction) {
        return accu(square(target - prediction))/target.n_rows;
    }
    
    //Backward step methods
    arma::mat outputGradient(arma::mat prediction) {
        return 2.0*(prediction-target)/target.n_rows;
    }
    
    std::vector<arma::mat> backwardGradients(std::vector<arma::mat> contextOverTime, arma::mat gradientOutput) {
        std::vector<arma::mat> cGOT, iGOT;
        arma::mat lastContextGradient = gradientOutput*outputWeights.t();
        arma::mat lastInputGradient = lastContextGradient*inputWeights.t();
//        std::cout << size(gradientOutput) << std::endl;
//        std::cout << size(outputWeights) << std::endl;
//        std::cout << size(input.col(0)) << std::endl;
//        std::cout << size(inputWeights) << std::endl;
//        std::cout << size(contextWeights) << std::endl;
//        std::cout << size(lastContextGradient) << std::endl;
//        std::cout << size(lastInputGradient*input.col(input.n_cols-1)) << std::endl;
//        std::cout << size(sum(lastContextGradient*input.col(input.n_cols-1), 1)) << std::endl;
        
        cGOT.push_back(lastContextGradient);
        iGOT.push_back(lastInputGradient);
        
        arma::mat inputGradientAccumulator;
        inputGradientAccumulator.zeros( size(inputWeights) );
        arma::mat contextGradientAccumulator;
        contextGradientAccumulator.zeros( size(contextWeights) );
        
        for(int i=0; i<input.n_cols; i++) {
//            std::cout << "step 1" << std::endl;
            inputGradientAccumulator += sum(iGOT[i]%input.col(input.n_cols-i-1));
            std::cout << "step 2" << std::endl;
            contextGradientAccumulator += sum(cGOT[i]*contextOverTime[input.n_cols-i-1]);
            std::cout << "step 3" << std::endl;
            cGOT.push_back(cGOT[i]*contextWeights);
            iGOT.push_back(iGOT[i]*inputWeights);
            std::cout << "step 4" << std::endl;
        }
        
        std::vector<arma::mat> retVec;
        retVec.push_back(inputGradientAccumulator);
        retVec.push_back(contextGradientAccumulator);
        
        return retVec;
    }
    
    //Model update step
    double updateModel() {
        double error = 0.0;
//        std::cout << "step 1" << std::endl;
        std::vector<arma::mat> cOT = contextOverTime();
//        std::cout << "step 2" << std::endl;
        arma::mat prediction = cOT.back()*outputWeights;
        arma::mat gradientOutput = outputGradient(prediction);
//        std::cout << "step 3" << std::endl;
        std::vector<arma::mat> weightGradients = backwardGradients(cOT, gradientOutput);
        std::cout << "step 4" << std::endl;
        return error;
    }
    
    //placeholder
//    double updateModel() {
//        double error = 0.0;
//        return error;
//    }
    
    public:
    arma::mat input;
    arma::mat target;
    
    RecurrentNeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    }
    
    std::vector<double> train(int numIt) {
        std::vector<double> costVector;
        for(int i=0; i<numIt; i++) {
            double cost = updateModel();
            costVector.push_back(cost);
        }
        return costVector;
    }
};
#endif
