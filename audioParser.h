#ifndef __AUDIOPARSER_H__
#define __AUDIOPARSER_H__
#include <sndfile.hh>
#include <armadillo>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#define PI 3.14159265

class AudioParser {
    arma::mat buffer;
    SF_INFO soundInfo;
    arma::cube frequencyMatrix;

    public:
    AudioParser() {
        buffer.zeros(1, 1);
        frequencyMatrix.zeros(1, 1, 1);
    }

    void readWaveFile(std::string fileName) {
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
            std::vector<double> tempBuffer(numElems);
            buffer.zeros(numFrames, numChannels);
            double numParsedFrames = sf_readf_double(sndFile, &tempBuffer[0], soundInfo.frames);
            if (numParsedFrames != numFrames) {
                fprintf(stderr, "Did not read enough frames for source\n");
            }
            for(long i=0; i<numElems; i++) {
                long row = i / numChannels;
                long column = i % numChannels;
                buffer(row,column) = tempBuffer[i];
            }
            sf_close(sndFile);
        }
    }

    void parseFrequencyStrengths(double overlap, long fftLength) {
        long numSamples = buffer.n_rows;
        if (numSamples > 1) {
            long numChannels = buffer.n_cols;

            double numIntervals = 0;
            if (numSamples%fftLength == 0) {
                numIntervals = (floor(numSamples/fftLength)-1)/overlap+1;
            }
            else {
                numIntervals = floor(numSamples/fftLength)/overlap+1;
            }
            long numSamplesOverlap = fftLength*overlap;
            long validFFTLength = fftLength/2;

            frequencyMatrix.zeros(numChannels, numIntervals, validFFTLength);

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

                    arma::cx_vec chunk = arma::conv_to<arma::cx_vec>::from(buffer.submat(fftRange, channelVec));

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
                    chunk = fft(chunk);

                    //drop frequencies below nyquist frequency
                    chunk.shed_rows(validFFTLength, fftLength-1);

                    //equalize frequency strengths for different fft lengths
                    chunk /= numAvailableSamples;

                    //apply absolute value, combines R & I components
                    arma::vec magnitudeChunk = abs(chunk);

                    //converts magnitude to dB scale
                    magnitudeChunk = 20.0*log10(magnitudeChunk);

                    for (double k=0; k<validFFTLength; k++) {
                        frequencyMatrix(i, j, k) = magnitudeChunk(k);
                    }
                }
            }

            //replace -inf with lowest valid value
            double minStrength = frequencyMatrix.elem(find_finite(frequencyMatrix)).min();
            frequencyMatrix.elem( find_nonfinite(frequencyMatrix) ).fill(minStrength);
        }
        else {
            std::cout << "No audio data available" << std::endl;
        }
    }

    void plotSpectrogram(bool smoothGraph) {
        long numIntervals = frequencyMatrix.n_cols;
        if (numIntervals > 1) {
            long frames = soundInfo.frames;
            long samplerate = soundInfo.samplerate;

            FILE *pipe = popen("gnuplot -persist" , "w");

            if (pipe != NULL) {

                double duration = (double) frames/samplerate;
                double maxFrequency = samplerate/2.0;
                int axisFontSize = 12;
                long numChannels = frequencyMatrix.n_rows;
                long numIntervals = frequencyMatrix.n_cols;
                long numFrequencyBins = frequencyMatrix.n_slices;

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

                fprintf(pipe, "%f %d %f\n", 0.0, 0, frequencyMatrix.min());
                for (long i=0; i<numIntervals; i++) {
                    double timePoint = (double) (i+1)/numIntervals*duration;
                    for (long j=0; j<numFrequencyBins; j++) {
                        double frequency = (double) j/numFrequencyBins*maxFrequency;
                        fprintf(pipe, "%f %f %f\n", timePoint, frequency, frequencyMatrix(0, i, j));
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
        else {
            std::cout << "No frequency strengths available" << std::endl;
        }
    }
};
#endif
