#ifndef __AUDIOPARSER_H__
#define __AUDIOPARSER_H__
#include <sndfile.hh>
#include <armadillo>
#define PI 3.14159265

class AudioParser {
    bool isBufferCreated = false;
    bool isFrequencyMappingCreated = false;
    SF_INFO soundInfo;

    public:
    //raw frame X channel
    arma::mat buffer;
    //chunk X bin of frequency strength 
    std::vector<arma::mat> frequencyMapping;

    AudioParser() {}

    void readWaveFile(std::string fileName) {
        //associate file to sound file reader
        SNDFILE *sndFile = sf_open(fileName.c_str(), SFM_READ, &soundInfo);

        if (sndFile == NULL) {
            //verify file readabilitiy
            fprintf(stderr, "Error reading source file '%s': %s\n", fileName.c_str(), sf_strerror(sndFile));
        }
        else if (soundInfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)) {
            //verify file type
            fprintf(stderr, "Input should be 16-bit wave file\n");
            sf_close(sndFile);
        }
        else {
            long numFrames = soundInfo.frames;
            long numChannels = soundInfo.channels;
            long numElems = numFrames*numChannels;
            
            //create temporary data store 
            std::vector<double> tempBuffer(numElems);
            
            buffer.zeros(numFrames, numChannels);
            
            //store frames as doubles
            double numParsedFrames = sf_readf_double(sndFile, &tempBuffer[0], soundInfo.frames);
            if (numParsedFrames != numFrames) {
                fprintf(stderr, "Did not read enough frames for source\n");
            }
            for(long i=0; i<numElems; i++) {
                long row = i / numChannels;
                long column = i % numChannels;
                
                //row is the frame, column is the channel
                buffer(row,column) = tempBuffer[i];
            }
            isBufferCreated = true;
            sf_close(sndFile);
        }
    }

    void parseFrequencyStrengths(double overlap, long fftLength) {
        if (isBufferCreated) {
            long numSamples = buffer.n_rows;
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

            for (int i=0; i<numChannels; i++) {
                arma::mat channelMat(numIntervals, validFFTLength, arma::fill::zeros);
                frequencyMapping.push_back(channelMat);
            }

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
                    chunk /= validFFTLength;

                    //apply absolute value, combines R & I components
                    arma::vec magnitudeChunk = abs(chunk);
                    
                    //Not Necessary ATM
                    //converts magnitude to dB scale
                    //magnitudeChunk = 20.0*log10(magnitudeChunk);

                    //replace -inf with lowest valid value
                    double minStrength = magnitudeChunk.elem(find_finite(magnitudeChunk)).min();
                    magnitudeChunk.elem( find_nonfinite(magnitudeChunk) ).fill(minStrength);

                    //i - channel, j - interval, k - frequency bin
                    for (double k=0; k<validFFTLength; k++) {
                        frequencyMapping[i](j, k) = magnitudeChunk(k);
                    }
                }
            }

            isFrequencyMappingCreated = true;
        }
        else {
            std::cout << "No audio data available" << std::endl;
        }
        
//        std::cout << "# of Raw frames: " << buffer.n_rows << std::endl;
//        std::cout << "FFT Length: " << fftLength << std::endl;
//        std::cout << "# of FFT Frames: " << frequencyMapping[0].n_rows << std::endl;
//        std::cout << "# of FFT Bins per Frame: " << frequencyMapping[0].n_cols << std::endl;
    }

    void plotSpectrogram(bool smoothGraph) {
        if (isFrequencyMappingCreated) {
            long frames = soundInfo.frames;
            long samplerate = soundInfo.samplerate;

            FILE *pipe = popen("gnuplot -persist" , "w");

            if (pipe != NULL) {

                double duration = (double) frames/samplerate;
                double maxFrequency = samplerate/2.0;
                int axisFontSize = 12;
                long numIntervals = frequencyMapping[0].n_rows;
                long numFrequencyBins = frequencyMapping[0].n_cols;

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

                fprintf(pipe, "%f %d %f\n", 0.0, 0, frequencyMapping[0].min());
                for (long i=0; i<numIntervals; i++) {
                    double timePoint = (double) (i+1)/numIntervals*duration;
                    for (long j=0; j<numFrequencyBins; j++) {
                        double frequency = (double) j/numFrequencyBins*maxFrequency;
                        fprintf(pipe, "%f %f %f\n", timePoint, frequency, frequencyMapping[0](i, j));
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

    double durationInSeconds() {
        if (isBufferCreated) {
            return soundInfo.frames / ((double) soundInfo.samplerate);
        }
        else {
            return 0;
        }
    }

    long numberOfFrames() {
        if (isBufferCreated) {
            return soundInfo.frames;
        }
        else {
            return 0;
        }
    }

    long sampleRate() {
        if (isBufferCreated) {
            return soundInfo.samplerate;
        }
        else {
            return 0;
        }
    }

};
#endif
