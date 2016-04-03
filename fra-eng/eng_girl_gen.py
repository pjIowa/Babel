import os

with open('eng.txt') as f:
    for line in f:
        i = line.replace("\n", "")
        os.system('say -o %s.wave -v samantha %s ' % (i, i))