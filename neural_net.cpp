#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <armadillo>

class Neuron {
    public:
    arma::mat weights;

    Neuron(int neuronCount, int inputsPerNeuron) {
        arma::arma_rng::set_seed(1);
        weights.randu(neuronCount, inputsPerNeuron);
        weights.elem( find(weights > 1.0) ).ones();
        weights.elem( find(weights < -1.0) ).fill(-1.0);
    }
};

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    Neuron L1;
    Neuron L2;
    int L1_node_count = 4;

    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights() {
        L1 = Neuron(L1_node_count, input.n_cols);
        L2 = Neuron(target.n_rows, L1_node_count);
    }

    public:
    std::vector<arma::mat> make_predictions(arma::mat x) {
        arma::mat L1_output = sigmoid(sum(x.each_row()%L1_weights, 1));
        arma::mat L2_output = sigmoid(sum(L1_output.each_row()%L2_weights, 1));
        return std::vector<arma::mat> { L1_node_count, L2_output };
    }

    NeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    }

    void train(int numIt) {
        for(int i=0; i<numIt; i++) {
            std::pair<arma::mat, arma::mat> predictions = make_predictions(input);
            arma::mat L1_error = predictions.first - target;
            arma::mat L2_error = predictions.second - target;
            arma::mat L1_adjustment = sum(input.each_col()%(L1_error%sigmoid_derivative(predictions.first)), 0);
            arma::mat L2_adjustment = sum(input.each_col()%(L2_error%sigmoid_derivative(predictions.second)), 0);
            weights -= adjustment;
        }
    }
};


int main() {
    arma::mat input = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    arma::mat target = {{0, 1, 1, 1, 1, 0, 0}};
    int numIterations = 60000;

    NeuralNetwork model(input, target.t());
    model.train(numIterations);
    //arma::mat test ={{1, 0, 0}};
    //std::cout << model.make_predictions(test) << std::endl;
    return 0;
}
