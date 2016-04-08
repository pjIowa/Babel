import os
from string import punctuation

with open('fra.txt') as f:
    for i,line in enumerate(f, 1):
        phrase = ''.join(c for c in line if c not in punctuation).strip()
        # wav file w/ little endian 16-bit integer depth @ 44100 Hz sample rate
        os.system('say -o %s.wav -v amelie %s --data-format=LEI16@44100' % (i, phrase))