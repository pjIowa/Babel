#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <armadillo>

class NeuralNetwork {
    arma::mat input;
    arma::mat target;
    arma::mat weights;

    arma::mat sigmoid(arma::mat x) {
        return 1.0 / (1.0 + arma::exp(-1.0*x));
    }

    arma::mat sigmoid_derivative(arma::mat x) {
        return x % (1-x);
    }

    void randomInitWeights() {
        arma::arma_rng::set_seed(1);
        weights.randu(1, input.n_cols);
        weights.elem( find(weights > 1.0) ).ones();
        weights.elem( find(weights < -1.0) ).fill(-1.0);
    }

    public:
    arma::mat make_predictions(arma::mat x) {
        return sigmoid(sum(x.each_row()%weights, 1));
    }

    NeuralNetwork(arma::mat i, arma::mat t) {
        input = i;
        target = t;
        randomInitWeights();
    }

    void train(int numIt) {
        for(int i=0; i<numIt; i++) {
            arma::mat predictions = make_predictions(input);
            arma::mat error = predictions - target;
            arma::mat adjustment = sum(input.each_col()%(error%sigmoid_derivative(predictions)), 0);
            weights -= adjustment;
        }
    }
};


int main() {
    arma::mat input = {{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};
    arma::mat target = {{0, 1, 1, 0}};
    int numIterations = 10000;

    NeuralNetwork model(input, target.t());
    model.train(numIterations);
    arma::mat test ={{1, 0, 0}};
    std::cout << model.make_predictions(test) << std::endl;
    return 0;
}
