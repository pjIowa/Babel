#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__
#include <armadillo>

class Layer {
    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }

    public:
    arma::mat weights;
    Layer() {
        weights.randu(1, 1);
    }

    Layer(int neuronCount, int inputsPerNeuron) {
        arma::arma_rng::set_seed(1);
        weights.randu(inputsPerNeuron, neuronCount);
    }
    
    arma::mat calculateOutput(arma::mat x) {
        return sigmoid(x*weights);
    }
};

class RecurrentNeuralNetwork {
    Layer L_xh;
    Layer L_hh;
    Layer L_ctx;
    Layer L_hy;
    
    
    int contextNodeCount = 4;
    
    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }
    
    void randomInitWeights() {
        //TODO: add bias to hidden and input layers
        L_xh  = Layer(contextNodeCount, input.n_cols);
        L_hh  = Layer(target.n_rows, target.n_rows);
        L_hy  = Layer(target.n_cols, contextNodeCount);
        L_ctx = Layer(contextNodeCount, target.n_rows );
    }
    
    public:
    arma::mat input;
    arma::mat target;
    
    RecurrentNeuralNetwork(arma::mat i, arma::mat t) {
        input = i.t();
        target = t.t();
        randomInitWeights();
    }
    
    std::vector<arma::mat> forwardStep(arma::mat x) {
        arma::mat L_x_forward = L_xh.calculateOutput(x);
        arma::mat L_ctx_forward = L_ctx.calculateOutput(L_hh.weights);
        L_ctx.weights = L_x_forward + L_ctx_forward;
        return std::vector<arma::mat> { L_x_forward, L_ctx_forward, L_hy.calculateOutput(L_ctx.weights) };
    }
    
    void train(int numIt) {
        std::vector<arma::mat> layerOutputs = forwardStep(input);
        arma::mat L_x_output = layerOutputs[0];
        arma::mat L_ctx_output = layerOutputs[1];
        arma::mat L_hy_output = layerOutputs[2];
        arma::mat targetError = target-L_hy_output;
        
    }
};
#endif
