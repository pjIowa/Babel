#include <iostream>
#include <fstream>
#include <armadillo>
#include <utility>
#include <string>
#include <sndfile.hh>
#include <cmath>
#include <math.h>
#include <fftw3.h>

#define PI 3.14159265

std::pair<arma::mat, SF_INFO>  readWaveFile(std::string fileName);
arma::cube parseFrequencyStrengths(arma::mat rawSound, double overlap, double fftLength);
void plotSpectrogram(arma::cube strengths, long frames, long samplerate);

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
        if (tempBuffer == NULL) {
            fprintf(stderr, "Could not allocate memory for file\n");
        }
        else {
            buffer.zeros(numFrames, numChannels);
            double numParsedFrames = sf_readf_double(sndFile, tempBuffer, soundInfo.frames);
            if (numParsedFrames != numFrames) {
                fprintf(stderr, "Did not read enough frames for source\n");
            }
            for(long i=0; i<numElems; i++) {
                long row = i / numChannels;
                long column = i % numChannels;
                buffer(row,column) = tempBuffer[i];
            }
            delete[] tempBuffer;
//            free(tempBuffer);
        }
        sf_close(sndFile);
    }
    return std::make_pair(buffer, soundInfo);
}

arma::cube parseFrequencyStrengths(arma::mat rawSound, double overlap, long fftLength) {
    long numSamples = rawSound.n_rows;
    long numChannels = rawSound.n_cols;
    
    double numIntervals = 0;
    if (numSamples%fftLength == 0) {
        numIntervals = (floor(numSamples/fftLength)-1)/overlap+1;
    }
    else {
        numIntervals = floor(numSamples/fftLength)/overlap+1;
    }
    long numSamplesOverlap = fftLength*overlap;
    long validFFTLength = fftLength/2;
    
    arma::cube retMatrix;
    retMatrix.zeros(numChannels, numIntervals, validFFTLength);
    
    for(double j=0; j<numIntervals; j++) {
        long startIndex = j*numSamplesOverlap;
        long endIndex = startIndex + fftLength - 1;
        if(endIndex >= numSamples) {
            endIndex = numSamples-1;
        }
        long numAvailableSamples = endIndex-startIndex+1;
        arma::uvec fftRange = arma::linspace<arma::uvec>(startIndex, endIndex, numAvailableSamples);
        
        for(double i=0; i<numChannels; i++) {
            arma::uvec channelVec;
            channelVec << i;
            
            arma::cx_vec chunk(numAvailableSamples);
            fftw_complex* in = reinterpret_cast<fftw_complex*> (chunk.colptr(0));
            fftw_plan plan = fftw_plan_dft_1d(numAvailableSamples, in, in, FFTW_FORWARD, FFTW_MEASURE);
            chunk = arma::conv_to<arma::cx_vec>::from(rawSound.submat(fftRange, channelVec));
            
            //hann window function, reduce spectral leakage
            for (double k=0; k<numAvailableSamples; k++) {
                double multiplier = 0.5 * (1 - cos(2.0*PI*k/(numAvailableSamples-1)));
                chunk(k) = multiplier * chunk(k);
            }
            
            if (numAvailableSamples < fftLength) {
                //zero fill remainder
                chunk.insert_rows(numAvailableSamples, fftLength-numAvailableSamples);
            }
            
            //compute FFT
            fftw_execute(plan);
            
            //drop frequencies below nyquist frequency
            chunk.shed_rows(validFFTLength, fftLength-1);
            
            //equalize frequency strengths for different fft lengths
            chunk /= numAvailableSamples;
            
            //apply absolute value, combines R & I components
            arma::vec magnitudeChunk = abs(chunk);
            
            //converts magnitude to dB scale
            magnitudeChunk = 20.0*log10(magnitudeChunk);
            
            for (double k=0; k<validFFTLength; k++) {
                retMatrix(i, j, k) = magnitudeChunk(k);
            }
        }
        
    }
    
    //replace -inf with lowest valid value
    double minStrength = retMatrix.elem(find_finite(retMatrix)).min();
    retMatrix.elem( find_nonfinite(retMatrix) ).fill(minStrength);
    
    return retMatrix;
}

void plotSpectrogram(arma::cube strengths, long frames, long samplerate, bool smoothGraph) {
    FILE *pipe = popen("gnuplot -persist" , "w");
    
    if (pipe != NULL) {
        
        double duration = (double) frames/samplerate;
        double maxFrequency = samplerate/2.0;
        int axisFontSize = 12;
        long numChannels = strengths.n_rows;
        long numIntervals = strengths.n_cols;
        long numFrequencyBins = strengths.n_slices;
        
        fprintf(pipe, "set view map\n");
        fprintf(pipe, "set dgrid3d\n");
        if (smoothGraph == true ) {
            fprintf(pipe, "set pm3d interpolate 25,25\n");
        }
        fprintf(pipe, "set xlabel 'Time (s)' font 'Times-Roman, %d' offset 0,-2,0\n", axisFontSize);
        fprintf(pipe, "set ylabel 'Frequency (Hz)' font 'Times-Roman, %d' offset -2,0,0\n", axisFontSize);
        fprintf(pipe, "set yr [0:%f]\n", maxFrequency);
        fprintf(pipe, "set xr [0:%f]\n", duration);
        fprintf(pipe, "splot '-' with pm3d \n");
        
        fprintf(pipe, "%f %d %f\n", 0.0, 0, strengths.min());
        for (long i=0; i<numIntervals; i++) {
            double timePoint = (double) (i+1)/numIntervals*duration;
            for (long j=0; j<numFrequencyBins; j++) {
                double frequency = (double) j/numFrequencyBins*maxFrequency;
                fprintf(pipe, "%f %f %f\n", timePoint, frequency, strengths(0, i, j));
            }
        }
        fprintf(pipe, "e");
        
        fflush(pipe);
        pclose(pipe);
    }
    else {
        std::cout << "Could not open gnuplot pipe" << std::endl;
    }
}

int main (void) {
    std::pair<arma::mat, SF_INFO> waveData = readWaveFile("440_sine.wav");
    arma::mat buffer = waveData.first;
    SF_INFO soundInfo = waveData.second;
    
    long fftLength = 1024;
    double overlap = 0.5;
    arma::cube parsedStrengths = parseFrequencyStrengths(buffer, overlap, fftLength);
    
    bool smoothFlag = false;
    plotSpectrogram(parsedStrengths, soundInfo.frames, soundInfo.samplerate, smoothFlag);
    
    return 0;
}