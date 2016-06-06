#include "audioParser.h"

int main (void) {
    std::cout << std::endl;
    std::clock_t startTime;
    AudioParser parser = AudioParser();
    startTime = std::clock();
    parser.readWaveFile("2.wav");
//    std::cout << "Time to Read File: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    long fftLength = 1024;
    double overlap = 0.5;
    startTime = std::clock();
    parser.parseFrequencyStrengths(overlap, fftLength);
    
    
//    std::cout << "Time to Convert with FFT: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

//    std::cout << "File Duration (seconds): " << parser.durationInSeconds() << std::endl;
//    std::cout << std::endl;
//    Not Needed    
//    startTime = std::clock();
//    parser.plotSpectrogram(false);
//    std::cout << "Spectrogram Plot Time: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << std::endl;
    
    return 0;
}
