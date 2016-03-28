#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <armadillo>

class Neuron {
    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }

    public:
    arma::mat weights;
    Neuron() {
        weights.randu(1, 1);
    }

    Neuron(int neuronCount, int inputsPerNeuron) {
        arma::arma_rng::set_seed(1);
        weights.randu(inputsPerNeuron, neuronCount);
        weights.elem( find(weights > 1.0) ).ones();
        weights.elem( find(weights < -1.0) ).fill(-1.0);
    }

    arma::mat calculateOutput(arma::mat i) {
        return sigmoid(i*weights);
    }
};

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    Neuron L1;
    Neuron L2;
    int L1NodeCount = 4;

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights() {
        L1 = Neuron(L1NodeCount, input.n_cols);
        L2 = Neuron(target.n_cols, L1NodeCount);
    }

    public:
    std::vector<arma::mat> calculateLayerOutputs(arma::mat x) {
        arma::mat L1_output = L1.calculateOutput(x);
        arma::mat L2_output = L2.calculateOutput(L1_output);
        return std::vector<arma::mat> { L1_output, L2_output };
    }

    NeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    }

    void train(int numIt) {
        for(int i=0; i<numIt; i++) {
            std::vector<arma::mat> layerOutputs = calculateLayerOutputs(input);
            arma::mat L1Output = layerOutputs[0];
            arma::mat L2Output = layerOutputs[1];

            arma::mat L2Error = target-L2Output;
            if (i%6000==0) {
                std::cout << "Step " << i << ": " << sum(L2Error, 0) << std::endl;
            }
            arma::mat L2Delta = L2Error%sigmoid_derivative(L2Output);

            arma::mat L1Error = L2Delta*L2.weights.t();
            arma::mat L1Delta = L1Error%sigmoid_derivative(L1Output);

            arma::mat L1Adjustment = input.t()*L1Delta;
            arma::mat L2Adjustment = L1Output.t()*L2Delta;

            L1.weights += L1Adjustment;
            L2.weights += L2Adjustment;
        }
    }
};


int main() {
    arma::mat input = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    arma::mat target = {{0, 1, 1, 1, 1, 0, 0}};
    int numIterations = 60000;

    std::cout << "Neural Network trained on XOR examples" << std::endl;
    NeuralNetwork model(input, target.t());
    model.train(numIterations);
    return 0;
}
