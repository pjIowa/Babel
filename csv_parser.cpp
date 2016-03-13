#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

std::vector< std::vector<float> > parseCsv(std::string fileName);
void printMatrix(std::vector< std::vector<float> > matrix);

int main() {
    std::string fileName = "diabetes.csv";
//    std::vector< std::vector<float> > csvMatrix = parseCsv(fileName);
//    printMatrix(csvMatrix);
    return 0;
}

std::vector< std::vector<float> > parseCsv(std::string fileName) {
    std::ifstream inputStream(fileName);
    std::vector< std::vector<float> > matrix;

    std::string line;
    while(std::getline(inputStream,line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> row;
        while(std::getline(lineStream,cell,',')) {
            row.push_back(std::stof(cell));
        }
        matrix.push_back(row);
    }
    
    return matrix;
}

void printMatrix(std::vector< std::vector<float> > matrix) {
    for(int i=0; i<matrix.size(); i++) {
        for(int j=0; j<matrix[i].size(); j++) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}