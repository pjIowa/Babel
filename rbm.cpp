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
    arma::mat negativeVisibleProbabilities;
    arma::umat positiveHiddenStates;

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
    double exampleCount;

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights(int visibleNodeCount) {
        L1 = Neuron(L1NodeCount, visibleNodeCount);
    }

    public:

    RBM(arma::mat i) {
        input = join_rows(arma::ones<arma::mat>(i.n_rows, 1), i);
        randomInitWeights(i.n_cols);
        exampleCount = i.n_rows;
    }

    void train(int numIt) {
        double error;

        for(int i=0; i<numIt; i++) {

            arma::mat positiveAssociations = L1.calculatePositiveAssociations(input);
            arma::mat negativeAssociations = L1.calculateNegativeAssociations();

            L1.weights += learningRate * ((positiveAssociations-negativeAssociations) / exampleCount);

            if (i%1000 == 0) {
                error = accu(square(input - L1.negativeVisibleProbabilities));
                std::cout << "Step "<< i<<": "<< error << std::endl;
            }
        }
    }

    arma::umat run(arma::mat i) {
        arma::mat input = join_rows(arma::ones<arma::mat>(i.n_rows, 1), i);
        arma::mat positiveAssociations = L1.calculatePositiveAssociations(input);
        return L1.positiveHiddenStates.cols(1, L1.positiveHiddenStates.n_cols-1);
    }
};


int main() {
    arma::mat input = {{1,1,1,0,0,0},{1,0,1,0,0,0},{1,1,1,0,0,0},{0,0,1,1,1,0},{0,0,1,1,0,0},{0,0,1,1,1,0}};
    int numIterations = 7000;

    std::cout << "RBM trained on multiple user's movie preferences" << std::endl;
    RBM model(input);

    std::clock_t startTime;
    startTime = std::clock();
    model.train(numIterations);
    std::cout << "Training Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;
    arma::mat test = {{0,0,0,1,1,0}};
    std::cout << "Test User:" << std::endl;
    std::cout << "Likes 1st Harry Potter:\t" << test(0,0) << std::endl;
    std::cout << "Likes Avatar:\t\t" << test(0,1) << std::endl;
    std::cout << "Likes 3rd LOTR:\t\t" << test(0,2) << std::endl;
    std::cout << "Likes Gladiator:\t" << test(0,3) << std::endl;
    std::cout << "Likes Titanic:\t\t" << test(0,4) << std::endl;
    std::cout << "Likes Troll 2:\t\t" << test(0,5) << std::endl << std::endl;
    startTime = std::clock();
    arma::umat hiddenStates = model.run(test);
    std::cout << "Prediction Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "Hidden Neuron Activations:" << std::endl;
    std::cout << "Likes Oscar Winners:\t" << hiddenStates(0,0) << std::endl;
    std::cout << "Likes SciFi / Fantasy:\t" << hiddenStates(0,1) << std::endl;
    return 0;
}
