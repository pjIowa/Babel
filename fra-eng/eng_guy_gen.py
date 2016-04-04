import os
from string import punctuation

with open('eng.txt') as f:
    for line in f:
        i = ''.join(c for c in line if c not in punctuation).strip()
        os.system('say -v alex %s -o "%s.wave"' % (i, i))