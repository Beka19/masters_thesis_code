'''
Created on 03.06.2020

@author: Beka
'''

import numpy as np
import matplotlib.pyplot as plt
import os

#np.random.seed(0)

def rotation(theta, data, mean):
    """
    
    Die Funktion nimmt die Daten(data), rotiert um einen Winkel(theta) mit Rotation-Uhrsrung im Zentrum der Daten(data).
    
    @param theta: Theta ist der Rotationswinkel.
    @param data: Datanmatrix    
    @param mean: Mittelwerte von den Verteilungen
        
    @return: Es werden die rotierenden Daten zurückgegeben
    """
    theta = np.deg2rad(theta); rot = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    shift_b = np.asarray([[1, 0,  mean[0]], [0, 1,  mean[1]], [0, 0, 1]])
    shift_f = np.asarray([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
    data = np.dot(shift_b, np.dot(rot, np.dot(shift_f, data)))
    return data

def get_data_1D(mean, std, weights, size):
    
    data = list()
    for (label_idx, _) in enumerate(range(len(mean))):
        data.append(list())
        start = 0
        for (weight_idx, weight) in enumerate(weights[label_idx][:-1]):
            end = start + weight * size
            loc_data = np.random.normal(mean[label_idx][weight_idx], std[label_idx][weight_idx], int(end - start))
            data[label_idx].extend(loc_data)
            start = end
        end = size
        loc_data = np.random.normal(mean[label_idx][-1], std[label_idx][-1], int(end - start))#
        data[label_idx].extend(loc_data)
    
    data = np.asarray(data)
    
    return data

def get_data_2D(mean, std, theta, size):
    
    """
    
    Die Funktion erstellt normal Verteilungen mit bestimmten Mittelwert, Varianz und Datengröße.
    
    @param mean: Mittelwerte von den Verteilungen
    @param std: Standartabweichung von Verteilungen 
    @param theta: Theta ist der Rotationswinkel.
    @param size: Die größe der Datenpunkte
       
    
    @return: Es werden Verteungsdaten in eine Liste zurückgegeben
    
    """
    
    dataList=list()
    for idx in range(0, len(mean)):
        data = [np.random.normal(mean[idx][0], np.asarray(std[idx][0]) * 2, size), np.random.normal(mean[idx][1], np.asarray(std[idx][1]) * 2, size), np.ones((size))]
        data = rotation(theta[idx], data, mean[idx])
        dataList.append(data[:2, :])
    return dataList

def reorder_data(data):
    data = np.asarray(data)
    data0 = np.concatenate((data[0, :, :], np.zeros((1, data[0, :, :].shape[1]))), axis = 0)
    data1 = np.concatenate((data[1, :, :], np.ones((1, data[1, :, :].shape[1]))), axis = 0)
    data = np.concatenate((data0, data1), axis = 1).transpose()
    
    return data

def visualize(data, pts):
    if (data.shape[1] == 3):
        visualize_2D(data, pts)
    elif (data.shape[1] == 2):
        visualize_1D(data, pts)
    else:
        raise NotImplementedError

def visualize_2D(data, pts):
    
    """
    
    Die Funktion visualisiert Verteilungen aus der Liste der Verteilungsdaten und unklassifisierte Punkte mit bestimten Farben.
    
    @param data: Verteilungen
    @param pts: unklassifisierte Punkte
    @param data_colors: Farbe der Verteilungen
    @param pt_color: Farbe der unklassifisierten Punkte
    @param plot_idx = ID der Subplots
        
    """
    
    
    (_, axes) = plt.subplots(1, 1)
    
    for grp_idx in range(0, 2):
        loc_data = data[np.argwhere(data[:, (data.shape[1] - 1)] == grp_idx).squeeze(), :]
        for dim_idx in np.arange(0, (data.shape[1] - 1), 2):
            axes.scatter(loc_data[:, dim_idx], loc_data[:, dim_idx + 1], color = "blue" if (grp_idx == 0) else "red", alpha = 0.25)

    for dim_idx in np.arange(0, (data.shape[1] - 1), 2):
        axes.scatter(pts[:, dim_idx], pts[:, dim_idx + 1], color = "green", alpha = 0.25)

def visualize_1D(data, pts):
    (_, axes1) = plt.subplots(1, 1)
    
    for label_idx in np.unique(data[:, -1]):
        loc_data = data[np.argwhere(data[:, 1] == label_idx).squeeze(), 0]
        axes1.scatter(loc_data, np.random.permutation(np.arange(0, 1, 1/len(loc_data))), color = "blue" if (label_idx == 0) else "red", alpha = 0.25)
    
    axes1.scatter(pts[:, 0], np.random.permutation(np.arange(0, 1, 1/len(pts[:, 0]))), color = "green", alpha = 0.25)

    plt.show(block = True)

def gen_lda_prob_data(size = 500, pattern_idx = 1):
    
    """
    Erzeugt Daten für Auswertung mit ein-dimensionalem Raum
    """
    
    mean_pttn_1 = [[0.2, 0.75], [0.3]]
    mean_pttn_2 = [[0.5], [0.5]]
    mean_pttn_3 = [[0.25], [0.5]]
    
    std_pttn_1 = [[1/8, 1/8], [1/6]]
    std_pttn_2 = [[1/4], [1/16]]
    std_pttn_3 = [[1/16], [1/4]]
    
    weights_pttn_1 = [[0.8, 0.2], [1]]
    weights_pttn_2 = [[1], [1]]
    weights_pttn_3 = [[1], [1]]
    
    mean = [mean_pttn_1, mean_pttn_2, mean_pttn_3]
    std = [std_pttn_1, std_pttn_2, std_pttn_3]
    weights = [weights_pttn_1, weights_pttn_2, weights_pttn_3]
    
    mean = mean[pattern_idx]
    std = std[pattern_idx]
    weights = weights[pattern_idx]
    
    data = get_data_1D(mean, std, weights, size)
    data = np.expand_dims(data, axis = 1)
    
    return reorder_data(data)
    
def gen_ann_prob_data(size = 500, pattern_idx = 0):
    
    """
    Die Funktion erzeugt die Daten wenn overwrite ==True, wenn nicht lädt sie die Daten.
    
    """
    
    mean_pttn_1  = [[0, -2], [2, 5]]
    mean_pttn_2  = [[-2, 0], [2, 0]]
    mean_pttn_3  = [[-1, 3], [2, -2]]
    mean_pttn_4  = [[0, 0], [0, 0]]
    
    std_pttn_1   = [[2.5, 4], [4, 4]]
    std_pttn_2   = [[0.8, 1.5], [1.2, 2]]
    std_pttn_3   = [[4, 5], [3, 5]]
    std_pttn_4   = [[1, 4], [1, 4]]
    
    theta_pttn_1 = [15, 300]
    theta_pttn_2 = [90, 90]
    theta_pttn_3 = [320, 320]
    theta_pttn_4 = [45, 320]
    
    mean = [mean_pttn_1, mean_pttn_2, mean_pttn_3, mean_pttn_4]
    std = [std_pttn_1, std_pttn_2, std_pttn_3, std_pttn_4]
    theta = [theta_pttn_1, theta_pttn_2, theta_pttn_3, theta_pttn_4]
    
    mean = mean[pattern_idx]
    std = std[pattern_idx]
    theta = theta[pattern_idx]
    
    data = get_data_2D(mean, std, theta, size)
    
    return reorder_data(data)    
    
def gen_knn_prob_data(size = 500, pattern_idx = 1):
    
    """
    Erzeugt Daten für Auswertung mit ein-dimensionalem Raum
    """
    
    mean_pttn_1  = [[-1, 0], [1, 0]]
    mean_pttn_2  = [[-0.25, 0], [0.25, 0]]
    mean_pttn_2  = [[0, 0], [0, 0]]
    
    std_pttn_1   = [[1.5, 4], [1.5, 4]]
    std_pttn_2   = [[0.2, 3], [0.2, 3]]
    
    theta_pttn_1 = [45, 45]
    theta_pttn_2 = [135, 135]
    
    mean = [mean_pttn_1, mean_pttn_2]
    std = [std_pttn_1, std_pttn_2]
    theta = [theta_pttn_1, theta_pttn_2]
    
    mean = mean[pattern_idx]
    std = std[pattern_idx]
    theta = theta[pattern_idx]
    
    data = get_data_2D(mean, std, theta, size)
    
    return reorder_data(data)
    
def gen_pca_prob_data(size = 500, pattern_idx = 0, trash_cnt = 20):
    
    """
    
    """
    
    mean_pttn_1  = [[-5, 0], [5, 0]]
    std_pttn_1   = [[1.5, 2], [1.5, 2]]
    theta_pttn_1 = [0, 0]

    mean_pttn_2  = [[-0.05, 0], [0.05, 0]]
    std_pttn_2   = [[0.015, 0.02], [0.015, 0.02]]
    theta_pttn_2 = [0, 0]

    
    mean = [mean_pttn_1, mean_pttn_2]
    std = [std_pttn_1, std_pttn_2]
    theta = [theta_pttn_1, theta_pttn_2]
    
    mean = mean[pattern_idx]
    std = std[pattern_idx]
    theta = theta[pattern_idx]
    
    data = get_data_2D(mean, std, theta, size)
    
    mean_trash = [[0, 0], [0, 0]]
    std_trash = [[2, 2], [2, 2]]
    theta_trash = [0, 0]
    
    for trash_idx in range(trash_cnt):
        loc_trash_data = get_data_2D(mean_trash, std_trash, theta_trash, size)
        data[0] = np.concatenate((data[0], loc_trash_data[0]), axis = 0)
        data[1] = np.concatenate((data[1], loc_trash_data[1]), axis = 0)
    
    data = reorder_data(data)
    
    return data
    
def gen_ica_prob_data(size = 500, pattern_idx = 0):
    
    """
    
    """
    
    #===========================================================================
    # data0 = np.random.poisson(1, size)
    # data1 = np.random.poisson(10, size)
    #===========================================================================
    data0 = np.random.binomial(4, 0.85, size)
    data1 = np.random.binomial(13, 0.1, size)
    data0 = np.expand_dims(data0, axis = 0); data0 = np.expand_dims(data0, axis = 0)
    data1 = np.expand_dims(data1, axis = 0); data1 = np.expand_dims(data1, axis = 0)
    data = np.concatenate((data0, data1), axis = 0)
    
    plt.hist(data0[0, 0, :], 50, alpha = 0.5)
    plt.hist(data1[0, 0, :], 50, alpha = 0.5)
    plt.show(block = True)

    noise0 = np.random.random((int(np.asarray(data).shape[2]*2)))
    noise0 = np.random.binomial(4, 0.2, int(np.asarray(data).shape[2]*2))
    noise0 = np.reshape(noise0, (2, np.asarray(data).shape[2]))
    noise0 = np.expand_dims(noise0, axis = 1)
    
    noise1 = np.random.random((int(np.asarray(data).shape[2]*2)))
#    noise1 = np.random.binomial(11, 0.5, int(np.asarray(data).shape[2]*2))
    noise1 = np.reshape(noise1, (2, np.asarray(data).shape[2]))
    noise1 = np.expand_dims(noise1, axis = 1)
    
    data = np.concatenate((data, noise0, noise1), axis = 1)
    
    weights = [[0.3, 0.4, 0.3], [0.25, 0.4, 0.35], [0.45, 0.2, 0.35]]
    
    data2 = list()
    for grp_idx in range(2):
        data2.append(list())
        for dim_idx_0 in range(3):
            loc_data = np.zeros((5000))
            for dim_idx_1 in range(3):
                loc_data += data[grp_idx, dim_idx_1, :] * weights[dim_idx_0][dim_idx_1]
        
            data2[grp_idx].append(loc_data)
    
    data2 = np.asarray(data2)
    
    data = reorder_data(data)
    data2 = reorder_data(data2)
    
    return (data, data2)
       
    