import matplotlib.pyplot as plt
import numpy as np
import struct
import os

def openData():
    with open('train-labels-idx1-ubyte', 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    
    with open('train-images-idx3-ubyte', 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows, cols)
    
    return labels, images


labels, images = openData()

#statisitics and display example image
print labels.shape
print images.shape
plt.imshow(images[0], cmap='Greys_r')
plt.show()