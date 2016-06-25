import numpy as np
import struct
import os

class PictureClassifier:
    
    def __init__(self):
        self.loadData()
        
    def loadData(self):
        with open('train-labels-idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.labels = np.fromfile(flbl, dtype=np.int8)
    
        with open('train-images-idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.images = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols)
    
    def getDim(self):
        print self.labels.shape
        

c = PictureClassifier()
c.getDim()