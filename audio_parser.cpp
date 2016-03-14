#include <iostream>
#include <fstream>
//#include <armadillo>
#include <string>
//#include	<cstdio>
//#include	<cstring>
#include <sndfile.hh>

#define BUFFER_LEN 1024

void loadWaveFile(std::string fileName);

class WavData {
    public:
        short* data;
        unsigned long size;
        
        WavData() {
            data = NULL;
            size = 0;
        }
};

void readWaveFile(std::string fileName) {
    short buffer [BUFFER_LEN];
    
	SndfileHandle file = SndfileHandle(fileName.c_str());

	printf ("Opened file '%s'\n", fileName.c_str());
	printf ("Sample rate : %d\n", file.samplerate());
	printf ("Channels    : %d\n", file.channels());

	file.read(buffer, BUFFER_LEN);
}

int main (void) { 
    readWaveFile("440_sine.wav");
}