#include "audioParser.h"

int main (void) {
    std::clock_t startTime;
    AudioParser parser = AudioParser();
    startTime = std::clock();
    parser.readWaveFile("440_sine.wav");
    std::cout << "File Read Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;

    long fftLength = 1024;
    double overlap = 0.5;
    startTime = std::clock();
    parser.parseFrequencyStrengths(overlap, fftLength);
    std::cout << "FFT Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;

    bool smoothFlag = false;
    startTime = std::clock();
    parser.plotSpectrogram(smoothFlag);
    std::cout << "Plotting Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;

    return 0;
}
