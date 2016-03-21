#include <iostream>
#include <armadillo>

void getNumIntervals(long numSamples, long fftLength, double overlap);

void getNumIntervals(long numSamples, long fftLength, double overlap) {
    double numIntervals = 0;
    if (numSamples%fftLength == 0) {
        numIntervals = (floor(numSamples/fftLength)-1)/overlap+1;
    }
    else {
        numIntervals = floor(numSamples/fftLength)/overlap+1;
    }
//    std::cout << numIntervals << std::endl;
//    std::cout << fftLength*overlap << std::endl;
}

int main (void) {
    long numChannels = 2;
    long numIntervals = 4;
    long numAvailableSamples = 3;
    long fftLength = 5;
    long validFrequencyBins = 5;
    arma::cube A;
    A.zeros(numChannels, numIntervals, validFrequencyBins);
    
    
    for(long i=0; i<numIntervals; i++) {
        for(long j=0; j<numChannels; j++) {
            arma::cube chunk(1, 1, numAvailableSamples);
            chunk(0, 0, 0) = 1.0;
            chunk(0, 0, 1) = 2.0;
            chunk(0, 0, 2) = 3.0;
            chunk.insert_slices(numAvailableSamples, fftLength-numAvailableSamples, true);
            A.tube(j, i) = chunk;
        }
    }
//    getNumIntervals(5060, 1024, 0.5);
//    getNumIntervals(5060, 1024, 0.25);
//    
//    getNumIntervals(5060, 128, 0.5);
//    getNumIntervals(5060, 128, 0.25);
//    
//    getNumIntervals(2048, 1024, 0.5);
//    getNumIntervals(2048, 1024, 0.25);
//    
//    getNumIntervals(2048, 128, 0.5);
//    getNumIntervals(2048, 128, 0.25);
    
    return 0;
}