#include <iostream>
#include <fstream>
#include <armadillo>
#include <utility>
#include <string>
#include <sndfile.hh>
#include <cmath>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fftw3.h>

#define PI 3.14159265

std::pair<arma::mat, SF_INFO>  readWaveFile(std::string fileName);
arma::mat parseFrequencyStrengths(arma::mat rawSound);

std::pair<arma::mat, SF_INFO> readWaveFile(std::string fileName) {
    SF_INFO soundInfo;
    arma::mat buffer;
    SNDFILE *sndFile = sf_open(fileName.c_str(), SFM_READ, &soundInfo);
    
    if (sndFile == NULL) {
        fprintf(stderr, "Error reading source file '%s': %s\n", fileName.c_str(), sf_strerror(sndFile));
    }
    else if (soundInfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)) {
        fprintf(stderr, "Input should be 16-bit wave file\n");
        sf_close(sndFile);
    }
    else {
        long numFrames = soundInfo.frames;
        long numChannels = soundInfo.channels;
        long numElems = numFrames*numChannels;
        double* tempBuffer = new double[numElems];
        buffer.zeros(numFrames, numChannels);
        long numParsedFrames = sf_readf_double(sndFile, tempBuffer, soundInfo.frames);
        if (numParsedFrames != numFrames) {
            fprintf(stderr, "Did not read enough frames for source\n");
        }
        for(long i=0; i<numElems; i++) {
            long row = i / numChannels;
            long column = i % numChannels;
            buffer(row,column) = tempBuffer[i];
        }
        free(tempBuffer);
        sf_close(sndFile);
    }
    return std::make_pair(buffer, soundInfo);
}

arma::mat parseFrequencyStrengths(arma::mat rawSound) {
    //break into equal chunks of sample points, n should be power of 2
    //start count at 128, higher count is higher frequency resolution 
    //Andrew Ng uses 20ms in paper, so try up to 1024
    long n = 1024;
    long rows = rawSound.n_rows;
    long cols = rawSound.n_cols;
    long numIntervals = rows/n;
    
    //keep frequencies up to and including 20kHz
    //this is within the expected Nyquist frequency: 22.5kHz
    long frequencyCap = 20000;
    
    arma::mat retMatrix;
    retMatrix.zeros(numIntervals, frequencyCap);
    
    for(long i=0; i<=numIntervals; i++) {
        
        //try 50% overlap, retain information lost from window function
        long start = i*n;
        long end = (i+1)*n-1;
        if (i==numIntervals) {
            end = rows-1;
        }
        long fftLength = end-start+1;
        
        arma::uvec fftRange = arma::linspace<arma::uvec>(start, end, fftLength);
        for(double j=0; j<cols; j++) {
            arma::uvec colVec;
            colVec << j;
            arma::cx_mat chunk(fftLength, 1);
            
            //apply fft
            //try fftw library for speed
            fftw_complex* in = reinterpret_cast<fftw_complex*> (chunk.colptr(0));
            fftw_plan plan = fftw_plan_dft_1d(fftLength, in, in, FFTW_FORWARD, FFTW_MEASURE);
            
            chunk = arma::conv_to<arma::cx_mat>::from(rawSound.submat(fftRange, colVec));
            
            //apply hann window function, reduce spectral leakage
            for (long i=0; i<fftLength; i++) {
                double multiplier = 0.5 * (1 - cos(2*PI*i/(n-1)));
                chunk(i, 0) = multiplier * chunk(i, 0);
            }
            
            fftw_execute(plan);
            
            //filter frequencies below cap
            
            //scale by n, remove effect on magnitude from length of signal
            //apply absolute value, combines R & I components
            
            //apply 20*log, converts magnitude to dB scale
            //each chunk now provides accurate frequency strength over time
        }
    }
    
    return retMatrix;
}

int main (void) {
    
    //Create Spectrogram
    //get raw signal and sound information
    std::pair<arma::mat, SF_INFO> waveData = readWaveFile("440_sine.wav");
    arma::mat buffer = waveData.first;
    SF_INFO soundInfo = waveData.second;
    
    //get frequency strengths per time period
    arma::mat parsedStrengths = parseFrequencyStrengths(buffer);
    
    //map in gnuplot
    
    return 0;
}