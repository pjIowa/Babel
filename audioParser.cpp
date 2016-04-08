#include "audioParser.h"
#include "neuralNet.h"

int main (void) {
    std::clock_t startTime;
    AudioParser parser = AudioParser();
    startTime = std::clock();
    parser.readWaveFile("2.wav");
    std::cout << "File Read Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;

    long fftLength = 1024;
    double overlap = 0.5;
    startTime = std::clock();
    parser.parseFrequencyStrengths(overlap, fftLength);
    std::cout << "FFT Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;

    return 0;
}
