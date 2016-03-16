#include <iostream>
#include <fstream>
#include <armadillo>
#include <utility>
#include <string>
#include <sndfile.hh>
#include <cmath>

std::pair< arma::mat, double > readWaveFile(std::string fileName);
arma::mat makeFFT(arma::mat wave);
void plotWave(arma::mat wave, double sampleFrequency);

std::pair< arma::mat, double > readWaveFile(std::string fileName) {
    
    SF_INFO sndInfo;
    SNDFILE *sndFile = sf_open(fileName.c_str(), SFM_READ, &sndInfo);
    
    if (sndFile == NULL) {
        fprintf(stderr, "Error reading source file '%s': %s\n", fileName.c_str(), sf_strerror(sndFile));
        return std::make_pair(arma::mat(), 0.0);
    }
    if (sndInfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)) {
        fprintf(stderr, "Input should be 16-bit wave file\n");
        sf_close(sndFile);
        return std::make_pair(arma::mat(), 0.0);
    }
    
    int numberOfFrames = sndInfo.frames;
    double *buffer = new double[numberOfFrames];
    if (buffer == NULL) {
        fprintf(stderr, "Could not allocate memory for data\n");
        sf_close(sndFile);
        return std::make_pair(arma::mat(), 0.0);
    }
    
    long numFrames = sf_readf_double(sndFile, buffer, sndInfo.frames);
    if (numFrames != sndInfo.frames) {
        fprintf(stderr, "Did not read enough frames for source\n");
        sf_close(sndFile);
        free(buffer);
        return std::make_pair(arma::mat(), 0.0);
    }
    
    sf_close(sndFile);
    arma::mat rawSoundMatrix(buffer, numberOfFrames, 2, true, true);
    free(buffer);
    
    printf("Read %ld frames from %s, Sample rate: %d Hz, Length: %f seconds\n", numFrames, fileName.c_str(), sndInfo.samplerate, (float)numFrames/sndInfo.samplerate);
    
    return std::make_pair(rawSoundMatrix, sndInfo.samplerate);
}

arma::mat makeFFT(arma::mat wave) {
    //TODO: use both channels
    arma::cx_mat p_complex = arma::fft(wave.col(0));
    int n = p_complex.n_rows; 
    int nUniquePoints = ceil((n+1)/2.0);
    arma::mat p = abs(p_complex.rows(0,nUniquePoints-1));
    p /= double(n);
    p = square(p);
    
    if (n%2 > 0) {
        p.rows(1, nUniquePoints-1) *= 2;
    }
    else {
        p.rows(1, nUniquePoints-2) *= 2;
    }
    return p;
}

void plotWave(arma::mat wave, double sampleFrequency) {
    FILE *pipe = popen("gnuplot -persist" , "w");
    
    if (pipe != NULL) {
        arma::mat frequencyPower = makeFFT(wave);
        
        frequencyPower = 10*log10(frequencyPower);
        
        arma::mat frequencies = arma::linspace<arma::mat>(0, frequencyPower.n_rows, frequencyPower.n_rows)*sampleFrequency/wave.n_rows;
        
        fprintf(pipe, "set style line 5 lt rgb 'red' lw 3 pt 6 \n");
        fprintf(pipe, "plot '-' with linespoints ls 5 \n");
        
        for (int i=0; i<frequencyPower.n_rows; i++) {
            fprintf(pipe, "%lf %lf\n", frequencies(i)/1000, frequencyPower(i));
        }
        
        fprintf(pipe, "e");
        fprintf(pipe, "set terminal png crop \n");
        fprintf(pipe, "set output 'fft.png' \n");
        fprintf(pipe, "set xlabel 'Frequency [kHz]' \n");
        fprintf(pipe, "set ylabel 'Intensity [dB]' \n");
        fprintf(pipe, "unset key \n");
        fprintf(pipe, "replot \n");
        fprintf(pipe, "unset output \n");

        fflush(pipe);
        pclose(pipe);
    }
    else {
        std::cout << "Could not open gnuplot pipe" << std::endl;
    }

}

int main (void) {
    std::pair< arma::mat, double > dataPair = readWaveFile("440_sine.wav");
    arma::mat rawWave = dataPair.first;
    double sampleFrequency = dataPair.second;
    plotWave(rawWave, sampleFrequency);
    
    return 0;
}