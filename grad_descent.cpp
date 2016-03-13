#include <iostream>
#include <armadillo>
#include <string>

int main() {
    std::string fileName = "diabetes.csv";
    arma::mat X;
    X.load(fileName, arma::csv_ascii);
    return 0;
}
