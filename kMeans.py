'''
Created on Sep 6, 2020

@author: Beka
'''

import numpy as np

import matplotlib.pyplot as plt
import SimulationTest
import sklearn.cluster

def transform(data):
    for (dataIdx, _) in enumerate(data):
        data[dataIdx, 0] = data[dataIdx, 0]
        data[dataIdx, 1] = data[dataIdx, 1] -np.exp(np.abs(data[dataIdx, 0]) / 2)
    
    return data

def visualize(data0, data00, kmeans0, data1, data11, kmeans1):
    (fig0, axes0) = plt.subplots(1, 2)
    axes0[0].scatter(data0[0, 0], data0[0, 1], color = "red")
    axes0[0].scatter(data0[1, 0], data0[1, 1], color = "blue")
    axes0[0].set_xlim((-20, 20))
    axes0[0].set_ylim((-20, 20))
    
    idx0 = np.argwhere(kmeans0.labels_ == 0).squeeze()
    idx1 = np.argwhere(kmeans0.labels_ == 1).squeeze()
    axes0[1].scatter(data00[idx0, 0], data00[idx0, 1], color = "orange")
    axes0[1].scatter(data00[idx1, 0], data00[idx1, 1], color = "green")
    axes0[1].set_xlim((-20, 20))
    axes0[1].set_ylim((-20, 20))
    
    (fig1, axes1) = plt.subplots(1, 2)
    axes1[0].scatter(data1[0, 0], data1[0, 1], color = "red")
    axes1[0].scatter(data1[1, 0], data1[1, 1], color = "blue")
    axes1[0].set_xlim((-20, 20))
    axes1[0].set_ylim((-20, 20))
    
    idx0 = np.argwhere(kmeans1.labels_ == 0).squeeze()
    idx1 = np.argwhere(kmeans1.labels_ == 1).squeeze()
    axes1[1].scatter(data11[idx0, 0], data11[idx0, 1], color = "orange")
    axes1[1].scatter(data11[idx1, 0], data11[idx1, 1], color = "green")
    axes1[1].set_xlim((-20, 20))
    axes1[1].set_ylim((-20, 20))

def main():
    size = 5000

    mean = [[0, 11], [0, 0]]
    std = [[1.25, 0.5], [0.5, 0.75]]
    theta = [180, 0]
    
    data0 = SimulationTest.get_data_2D(mean, std, theta, size)
    data0 = np.asarray(data0)
    data00 = np.concatenate((data0[0], data0[1]), axis = 1).transpose()
    
    data1 = transform(np.copy(data0))
    data11 = np.concatenate((data1[0], data1[1]), axis = 1).transpose()
    
    kmeans0 = sklearn.cluster.KMeans(2).fit(data00)
    kmeans1 = sklearn.cluster.KMeans(2).fit(data11)
    
    visualize(data0, data00, kmeans0, data1, data11, kmeans1)

main()
plt.show(block = True)



