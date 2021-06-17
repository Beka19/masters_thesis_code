'''
Created on Aug 22, 2020

@author: voodoocode
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import SimulationTest as st

import sklearn.neighbors as ne

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np 

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def addData(data, mean, std, weights, size):
    
    locData = st.get_data_1D(mean, std, weights, size)
    locData = np.asarray(locData)
    locData = np.expand_dims(np.concatenate((locData[0], locData[1]), axis = 0), axis = 1)
    
    data = np.concatenate((locData, data), axis = 1)
    
    return data

size = 1000
badDims = 31

print("Size %i | bad dims %i" % (size, badDims))

data = np.zeros((int(size * 2), 0))

data = addData(data, [[5], [2]], [[2], [1]], [[1], [1]], size)
for x in range(badDims):
    mean = np.random.randint(0, 10, 1)
    data = addData(data, [[mean], [mean]], [[np.sqrt(mean)], [np.sqrt(mean)]], [[1], [1]], size)

data = np.concatenate((data, np.expand_dims(np.asarray([0] * size + [1] * size), axis = 1)), axis = 1)

randList = np.arange(0, size * 2)
randList = np.random.permutation(randList)

data = data[randList, :]

annScore = list()
knnScore = list()

testBlockSize = int(size * 2 / 10)

start = 0

for x in range(10):

    end = int((x + 1) * testBlockSize)

    train_data = np.concatenate((data[:start], data[end:]), axis = 0)
    test_data = data[start:end]

#    train_data = data[randList[int(size * 2 / 10):], :]
#    test_data = data[randList[:int(size * 2 / 10)], :]
    
    while(True):
    
        model = Sequential()
        model.add(Dense(16, input_dim=(train_data.shape[1] - 1), activation = "relu"))
        model.add(Dropout(.33))
        model.add(Dense( 8, activation = "relu"))
        model.add(Dropout(.33))
        model.add(Dense( 4, activation = "relu"))
        model.add(Dense( 2, activation = "relu"))
        model.add(Dense( 1, activation = "sigmoid"))
        
        model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
        model.fit(train_data[:, :(train_data.shape[1] - 1)], train_data[:, (train_data.shape[1] - 1)], batch_size=1, epochs = 10, verbose = 0)
        trainScore = model.evaluate(train_data[:, :(train_data.shape[1] - 1)], train_data[:, (train_data.shape[1] - 1)], verbose = 0)[1]
        
        if (trainScore > 0.6):
            break
    
    annScore.append(np.sum(model.predict(test_data[:, :(test_data.shape[1] - 1)]).round().squeeze()  == test_data[:, (test_data.shape[1] - 1)]) / test_data.shape[0])
    
    
    knn = ne.KNeighborsClassifier(n_neighbors = 5)
    knn.fit(train_data[:, :(train_data.shape[1] - 1)], train_data[:, (train_data.shape[1] - 1)])
    knnScore.append(np.sum(knn.predict(test_data[:, :(test_data.shape[1] - 1)]).squeeze()  == test_data[:, (test_data.shape[1] - 1)]) / test_data.shape[0])
    
    start = end
    
    print(annScore, knnScore)

annScore = np.mean(annScore)
knnScore = np.mean(knnScore)

print(annScore, knnScore)

plt.show(block = True)



