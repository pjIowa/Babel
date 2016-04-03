import os

with open('fra.txt') as f:
    for line in f:
        i = line.replace("\n", "")
        os.system('say -o %s.wave -v thomas %s ' % (i, i))