#include <iostream>
#include <fstream>
#include <armadillo>
#include <string>
#include <malloc/malloc.h>
#include <sndfile.hh>

arma::mat readWaveFile(std::string fileName);

arma::mat readWaveFile(std::string fileName) {
    
    SF_INFO sndInfo;
    SNDFILE *sndFile = sf_open(fileName.c_str(), SFM_READ, &sndInfo);
    
    if (sndFile == NULL) {
        fprintf(stderr, "Error reading source file '%s': %s\n", fileName.c_str(), sf_strerror(sndFile));
        return arma::mat();
    }
    if (sndInfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)) {
        fprintf(stderr, "Input should be 16-bit wave file\n");
        sf_close(sndFile);
        return arma::mat();
    }
    
    int numberOfFrames = sndInfo.frames;
    double *buffer = new double[numberOfFrames];
    if (buffer == NULL) {
        fprintf(stderr, "Could not allocate memory for data\n");
        sf_close(sndFile);
        return arma::mat();
    }
    
    long numFrames = sf_readf_double(sndFile, buffer, sndInfo.frames);
    if (numFrames != sndInfo.frames) {
        fprintf(stderr, "Did not read enough frames for source\n");
        sf_close(sndFile);
        free(buffer);
        return arma::mat();
    }
    
    sf_close(sndFile);
    arma::mat rawSoundMatrix(buffer, numberOfFrames, 1, true, true);
    free(buffer);
    
    printf("Read %ld frames from %s, Sample rate: %d Hz, Length: %f seconds\n", numFrames, fileName.c_str(), sndInfo.samplerate, (float)numFrames/sndInfo.samplerate);
    
    return rawSoundMatrix;
}

int main (void) { 
    arma::mat rawSound = readWaveFile("440_sine.wav");
    return 0;
}