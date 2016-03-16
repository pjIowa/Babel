from pylab import *
from scipy.io import wavfile
import os

def parseAudio():
    # int16, 1 int at 16 bit
    # sound pressure vs. time (ms)
    sampFreq, snd = wavfile.read('440_sine.wav')
    s1 = snd[:,0]
    plotFFT(sampFreq, s1)
    
    # complex128, 2 floats at 64 bit
    # magnitude and phase of frequencies
    p = fft(s1)
    n = len(s1) 
    nUniquePts = ceil((n+1)/2.0)
    
    # mean of signal and positive frequency mangitudes
    p = abs(p[0:nUniquePts])
    
    # scale magnitude by length of signal and square to get power J/s
    p /= float(n)
    p = p**2
    
    # keep only frequencies below nyquist frequency
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2
    
    # convert to power in dB
    p = 10*log10(p)
    
    # final frequency list
    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n)

    RANGE = [40,80,120,180,300]
    
    

# plot tone, amplitude vs. time (ms)
def plotTone(sampFreq, s1):
    timeArray = arange(0, 5060.0, 1)
    timeArray = timeArray / sampFreq
    timeArray = timeArray * 1000
    plot(timeArray, s1, color='k')
    ylabel('Amplitude')
    xlabel('Time (ms)')
    show()
    
# plot frequency (kHz) vs. power (dB)
def plotFFT(sampFreq, s1):
    n = len(s1) 
    p = fft(s1)
    nUniquePts = ceil((n+1)/2.0)
    p = p[0:nUniquePts]
    p = abs(p)
    p = p / float(n)
    p = p**2
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2
    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
    plot(freqArray/1000, 10*log10(p), color='k')
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')
    show()
    
def playAudio():
    os.system('afplay "440_sine.wav"')
            
parseAudio()
#playAudio()