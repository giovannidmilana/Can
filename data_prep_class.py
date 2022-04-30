import os
import numpy as np
from os import listdir
from numpy import savez_compressed
from numpy import load
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray

y, X = list(), list()

directory = 'seg_train'
i = -1
for folder in listdir(directory):
    i +=1
    print(folder)
    for file in listdir(directory + '/' + folder):
       #print(i)
       l = [0,0,0,0,0,0]
       l[i] = 1
       photo = load_img((directory + '/' + folder + '/' + file), target_size=(128, 128))
       photo = img_to_array(photo)
       X.append(photo)
       y.append(np.array(l))
       
       
X = asarray(X)
print(len(X))

y = asarray(y)

savez_compressed('EnvX128.npz', X)
savez_compressed('EnvY.npz', y)
