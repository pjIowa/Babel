from math import ceil, floor

def test(numSamplePoints, overlap, fftLength):
    startIndex = 0
    numIntervals = 0
    
    while startIndex < numSamplePoints:
        endIndex = startIndex + fftLength
        if endIndex >= numSamplePoints:
            endIndex = numSamplePoints - 1
#        print startIndex, endIndex
        numIntervals += 1
        if endIndex == (numSamplePoints-1):
            break
        else:
            startIndex += fftLength*overlap
#    print " " 
    print numIntervals
#    print numSamplePoints, " " , fftLength, " ", overlap
#    if numSamplePoints%fftLength == 0:
#        print (floor(numSamplePoints/fftLength)-1)/overlap +1
#    else:
#        print floor(numSamplePoints/fftLength)/overlap + 1
    

test(5060.0, 0.5, 1024.0)
test(5060.0, 0.25, 1024.0)

test(5096.0, 0.5, 128.0)
test(5096.0, 0.25, 128.0)

test(2048.0, 0.5, 1024.0)
test(2048.0, 0.25, 1024.0)

test(2048.0, 0.5, 128.0)
test(2048.0, 0.25, 128.0)