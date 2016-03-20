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
arma::cube parseFrequencyStrengths(arma::mat rawSound);
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
            free(tempBuffer);
        }
        sf_close(sndFile);
    }
    return std::make_pair(buffer, soundInfo);
}

arma::cube parseFrequencyStrengths(arma::mat rawSound) {
    double numFrequencyBins = 1024;
    double validFrequencyBins = floor(numFrequencyBins/2.0);
    double numSamples = rawSound.n_rows;
    double numChannels = rawSound.n_cols;
    double numIntervals = floor(numSamples/numFrequencyBins)+1;
    
    arma::cube retMatrix;
    retMatrix.zeros(numChannels, numIntervals, validFrequencyBins);
    
    for(double i=0; i<numIntervals; i++) {
        //try 50% overlap, retain information lost from window function
        double start = i*numFrequencyBins;
        double end = (i+1)*numFrequencyBins-1;
        if (i==(numIntervals-1)) {
            end = numSamples-1;
        }
        double fftLength = end-start+1;
        arma::uvec fftRange = arma::linspace<arma::uvec>(start, end, fftLength);
        
        for(double j=0; j<numChannels; j++) {
            arma::uvec colVec;
            colVec << j;
            arma::cx_mat chunk(fftLength, 1);
            
            //compute FFT
            fftw_complex* in = reinterpret_cast<fftw_complex*> (chunk.colptr(0));
            fftw_plan plan = fftw_plan_dft_1d(fftLength, in, in, FFTW_FORWARD, FFTW_MEASURE);
            chunk = arma::conv_to<arma::cx_mat>::from(rawSound.submat(fftRange, colVec));
            //hann window function, reduce spectral leakage
            for (double k=0; k<fftLength; k++) {
                double multiplier = 0.5 * (1 - cos(2.0*PI*k/(fftLength-1)));
                chunk(k, 0) = multiplier * chunk(k, 0);
            }
            //zero fill
            chunk.insert_rows(fftLength, numFrequencyBins-fftLength, true);
            fftw_execute(plan);
            
            //drop frequencies below nyquist frequency
            chunk.shed_rows(validFrequencyBins, numFrequencyBins-1);
            
            //equalize frequency strengths for different fft lengths
            chunk /= fftLength;
            
            //apply absolute value, combines R & I components
            arma::mat magnitudeChunk = abs(chunk);
            
            //converts magnitude to dB scale
            magnitudeChunk = 20.0*log10(magnitudeChunk);
            
            for (double k=0; k<validFrequencyBins; k++) {
                retMatrix(j, i, k) = magnitudeChunk(k, 0);
            }
        }
    }
    
    //replace -inf with lowest valid value
    double minStrength = retMatrix.elem(find_finite(retMatrix)).min();
    retMatrix.elem( find_nonfinite(retMatrix) ).fill(minStrength);
    return retMatrix;
}

void plotSpectrogram(arma::cube strengths, long frames, long samplerate) {
    FILE *pipe = popen("gnuplot -persist" , "w");
    
    if (pipe != NULL) {
        
        double duration = (double) frames/samplerate;
        double maxFrequency = samplerate/2.0;
        long axisFontSize = 12;
        long numChannels = strengths.n_rows;
        long numIntervals = strengths.n_cols;
        long numFrequencyBins = strengths.n_slices;
        
        fprintf(pipe, "set view map\n");
        fprintf(pipe, "set dgrid3d\n");
        fprintf(pipe, "set pm3d interpolate 25,25\n");
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
    arma::cube parsedStrengths = parseFrequencyStrengths(buffer);
    plotSpectrogram(parsedStrengths, soundInfo.frames, soundInfo.samplerate);
    return 0;
}