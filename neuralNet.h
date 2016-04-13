#ifndef __NEURALNET_H__
#define __NEURALNET_H__
#include <vector>
#include <string>
#include <math.h>
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

    arma::mat calculateOutput(arma::mat i) {
        return sigmoid(i*weights);
    }
};

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    Layer L1;
    Layer L2;
    int L1NodeCount = 4;

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights() {
        L1 = Layer(L1NodeCount, input.n_cols);
        L2 = Layer(target.n_cols, L1NodeCount);
    }

    public:
    std::vector<arma::mat> calculateLayerOutputs(arma::mat x) {
        arma::mat L1_output = L1.calculateOutput(x);
        arma::mat L2_output = L2.calculateOutput(L1_output);
        return std::vector<arma::mat> { L1_output, L2_output };
    }

    NeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t.t();
        randomInitWeights();
    }

    std::vector<double> train(int numIt) {
        std::vector<double> costVector;
        for(int i=0; i<numIt; i++) {
            std::vector<arma::mat> layerOutputs = calculateLayerOutputs(input);
            arma::mat L1Output = layerOutputs[0];
            arma::mat L2Output = layerOutputs[1];

            arma::mat L2Error = target-L2Output;
            arma::mat L2Delta = L2Error%sigmoid_derivative(L2Output);

            arma::mat L1Error = L2Delta*L2.weights.t();
            arma::mat L1Delta = L1Error%sigmoid_derivative(L1Output);

            arma::mat L1Adjustment = input.t()*L1Delta;
            arma::mat L2Adjustment = L1Output.t()*L2Delta;

            L1.weights += L1Adjustment;
            L2.weights += L2Adjustment;
            
            costVector.push_back(accu(L2Error));
        }
        return costVector;
    }
};

#endif
