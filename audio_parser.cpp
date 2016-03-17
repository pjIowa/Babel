#include <iostream>
#include <fstream>
#include <armadillo>
#include <utility>
#include <string>
#include <sndfile.hh>
#include <cmath>
#include <vector>
#include <fftw3.h>

std::pair<std::vector<double>, SF_INFO>  readWaveFile(std::string fileName);

std::pair<std::vector<double>, SF_INFO> readWaveFile(std::string fileName) {
    SF_INFO soundInfo;
    std::vector<double> buffer = std::vector<double>();
    SNDFILE *sndFile = sf_open(fileName.c_str(), SFM_READ, &soundInfo);
    
    if (sndFile == NULL) {
        fprintf(stderr, "Error reading source file '%s': %s\n", fileName.c_str(), sf_strerror(sndFile));
    }
    else if (soundInfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)) {
        fprintf(stderr, "Input should be 16-bit wave file\n");
        sf_close(sndFile);
    }
    else {
        int numberOfFrames = soundInfo.frames;
        int numberOfChannels = soundInfo.channels;
        double *tempBuffer = new double[numberOfFrames*numberOfChannels];
        if (tempBuffer == NULL) {
            fprintf(stderr, "Could not allocate memory for data\n");
        }
        else {
            long numFrames = sf_readf_double(sndFile, tempBuffer, soundInfo.frames);
            if (numFrames != soundInfo.frames) {
                fprintf(stderr, "Did not read enough frames for source\n");
            }
            //TODO: copy data from tempBuffer into buffer
        }
        
        free(tempBuffer);
        sf_close(sndFile);
    }
    return std::make_pair(buffer, soundInfo);
}

int main (void) {
    
    //Create Spectrogram
    //get raw signal and sound information
    
    std::pair<std::vector<double>, SF_INFO> waveData = readWaveFile("440_sine.wav");
    std::vector<double> buffer = waveData.first;
    SF_INFO soundInfo = waveData.second;
    
    //apply hann window function, reduce spectral leakage
    //break into equal chunks of sample points, count should be power of 2
    //use 50% overlap, retain information lost from window function
    //start count at 128, higher count is higher frequency resolution 
    //Andrew Ng uses 20ms in paper, so also try 1024
    //for each chunk and channel
        //set n to number of sample points
        //determine correct window function and apply
        //apply fft, use fftw library for speed
        //keep all frequencies below nyquist frequency, sampleFrequency/2
        //if n is even, keep the nyquist frequency too
        //scale by n, remove effect on magnitude from length of signal
        //apply absolute value, combines R & I components
        //apply 20*log, converts magnitude to dB scale
        //each chunk now provides accurate frequency strength over time
    
    return 0;
}