#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <ctime>
#include <armadillo>

class Neuron {
    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + exp(-1.0*x));
    }

    public:
    arma::mat weights;
    arma::mat negativeVisibleProbabilities = { {1, 3, 5}, {2, 4, 6} };
    arma::umat positiveHiddenStates = { {1, 1, 1}, {0, 0, 0} };

    Neuron() {
        weights.randu(1, 1);
    }

    Neuron(int neuronCount, int inputsPerNeuron) {
        arma::arma_rng::set_seed(1);
        weights.randn(inputsPerNeuron, neuronCount);
        weights *= 0.1;
        weights.insert_rows(0, 1);
        weights.insert_cols(0, 1);
    }

    arma::mat calculatePositiveAssociations(arma::mat x) {
        arma::mat positiveHiddenActivations = x*weights;
        arma::mat positiveHiddenProbabilities = sigmoid(positiveHiddenActivations);
        arma::arma_rng::set_seed(1);
        arma::mat randomNormalProbabilties = arma::randu(size(positiveHiddenProbabilities));
        positiveHiddenStates = positiveHiddenProbabilities > randomNormalProbabilties;
        return x.t()*positiveHiddenProbabilities;
    }

    arma::mat calculateNegativeAssociations() {
        arma::mat negativeVisibleActivations = positiveHiddenStates*weights.t();
        negativeVisibleProbabilities = sigmoid(negativeVisibleActivations);
        negativeVisibleProbabilities.col(0) = arma::ones<arma::vec>(negativeVisibleProbabilities.n_rows);
        arma::mat negativeHiddenActivations = negativeVisibleActivations*weights;
        arma::mat negativeHiddenProbabilities = sigmoid(negativeHiddenActivations);
        return negativeVisibleProbabilities.t()*negativeHiddenProbabilities;
    }
};

class RBM {
    arma::mat input;
    Neuron L1;
    int L1NodeCount = 2;
    double learningRate = 0.0005;
    double exampleCount = 1;

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights(int visibleNodeCount) {
        L1 = Neuron(L1NodeCount, visibleNodeCount);
    }

    public:

    RBM(arma::mat i) {
        std::cout <<"Raw Input Size: " << size(i) << std::endl;
        input = join_rows(arma::ones<arma::mat>(i.n_rows, 1), i);
        randomInitWeights(i.n_cols);
        exampleCount = i.n_rows;
    }

    void train(int numIt) {
        double error = 1.0;

        for(int i=0; i<numIt; i++) {

            arma::mat positiveAssociations = L1.calculatePositiveAssociations(input);
            arma::mat negativeAssociations = L1.calculateNegativeAssociations();

            L1.weights += learningRate * ((positiveAssociations-negativeAssociations) / exampleCount);

            if (i%1000 == 0) {
                error = accu(square(input - L1.negativeVisibleProbabilities));
                std::cout << "Loss: "<< error << std::endl;
            }
        }
    }
};


int main() {
    //movie mapping: Harry Potter 1, Avatar, LOTR 3, Gladiator, Titanic, Troll 2
    arma::mat input = {{1,1,1,0,0,0},{1,0,1,0,0,0},{1,1,1,0,0,0},{0,0,1,1,1,0},{0,0,1,1,0,0},{0,0,1,1,1,0}};
    int numIterations = 7000;

    std::cout << "RBM trained on movie examples" << std::endl;
    RBM model(input);

    std::clock_t startTime;
    startTime = std::clock();
    model.train(numIterations);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    arma::mat test = {{0,0,0,1,1,0}};
    return 0;
}
