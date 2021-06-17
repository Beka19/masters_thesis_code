'''
Created on 21.06.2020

@author: Beka
'''
import SimulationTest as st

import sklearn.decomposition as decomp
import sklearn.discriminant_analysis as da
import sklearn.neighbors as ne

import numpy as np

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def applyPCA(data, pts):
    pca = decomp.PCA(whiten = True)
    pca.fit(data[:, :(data.shape[1] - 1)])
    
    pca_data = np.copy(data)
    pca_data[:, :(data.shape[1] - 1)] = pca.transform(data[:, :(data.shape[1] - 1)])
    
    if (pts is None):
        return (pca_data, pca.explained_variance_ratio_)
    
    pca_pts = np.copy(pts)
    pca_pts[:, :(data.shape[1] - 1)] = pca.transform(pts[:, :(data.shape[1] - 1)])
    
    return (pca_data, pca_pts, pca.explained_variance_ratio_)

def applyICA(data, pts):
    ica = decomp.FastICA(whiten = True)
    ica.fit(data[:, :(data.shape[1] - 1)], )
    
    ica_data = np.copy(data)
    ica_pts = np.copy(pts)
    ica_data[:, :(data.shape[1] - 1)] = ica.transform(data[:, :(data.shape[1] - 1)])
    ica_pts[:, :(data.shape[1] - 1)] = ica.transform(pts[:, :(data.shape[1] - 1)])
    
    return (ica_data, ica_pts)

def normalizeData(data):
    for dim_idx in range((data.shape[1] - 1)):
        data[:, dim_idx] -= np.min(data[:, dim_idx])
        data[:, dim_idx] /= np.max(data[:, dim_idx])
        data[:, dim_idx] -= 0.5
        data[:, dim_idx] *= 2
    
    data = data[np.random.permutation(np.arange(data.shape[0])), :]
    pts = data[:100, :]
    data = data[100:, :]

    return (data, pts)

def extract_pts(data):
    data = data[np.random.permutation(np.arange(data.shape[0])), :]
    pts = data[:100, :]
    data = data[100:, :]

    return (data, pts)

def applyLDA(data, pts):
    lda = da.LinearDiscriminantAnalysis()
    lda.fit(data[:, :(data.shape[1] - 1)], data[:, (data.shape[1] - 1)])
    pred_pts = lda.predict(pts[:, :(data.shape[1] - 1)])
    
    return (pred_pts)

def applyKNN(data, pts):
    knn = ne.KNeighborsClassifier(n_neighbors = 5)
    knn.fit(data[:, :(data.shape[1] - 1)], data[:, (data.shape[1] - 1)])
    pred_pts = knn.predict(pts[:, :(data.shape[1] - 1)])
    
    return pred_pts

def applyML(data, pts):
    lda_pred_pts = applyLDA(data, pts)
    acc_lda = np.sum(lda_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tLDA %2.2f" % (acc_lda, ))
    
    knn_pred_pts = applyKNN(data, pts)
    acc_knn = np.sum(knn_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tKNN %2.2f" % (acc_knn, ))
    
    acc_ann = applyANN(data, pts)
    print("\tANN %2.2f" % (acc_ann, ))
    
    return (acc_lda, acc_knn, acc_ann)

def applyANN(data, pts):
    dim = data.shape[1] - 1
    
    trainCnt = 0
    while(trainCnt < 3):
        #Alle Ebene werden nacheinander verarbeiten
        model = tf.keras.Sequential()
        #Fügen input Ebenen hinzu
        model.add(tf.keras.Input(shape=((data.shape[1] - 1),)))
        #Dense Schicht wird konfiguriert mit 64 Neuronen 
        if (dim > 1):
            model.add(tf.keras.layers.Dense(dim, activation="relu"))
    #    model.add(tf.keras.layers.Dense(128, activation="relu"))
    #    model.add(tf.keras.layers.Dense(64, activation="relu"))
    #    model.add(tf.keras.layers.Dense(8, activation="relu"))
        #neoch ein mal Dense Schicht mit einem Neuron
        model.add(tf.keras.layers.Dense(1, activation="relu", input_dim = 64))
    
        #Model erstelt und Optimierung wird konfiguriert
        model.compile(optimizer = "SGD", loss = "binary_crossentropy", metrics=['accuracy'])
        #Model wird auf meine Daten träiniert
        model.fit(data[:, :(data.shape[1] - 1)], data[:, (data.shape[1] - 1)], epochs = 10, batch_size = 100, verbose = 0)
    
        trainScore = model.evaluate(data[:, :(data.shape[1] - 1)], data[:, (data.shape[1] - 1)], verbose = 0)[1]
        
        if (trainScore > 0.6):
            break
        trainCnt += 1
    
    
    #Evaluiere ich gegen neue Daten
    acc = model.evaluate(pts[:, :(data.shape[1] - 1)], pts[:, (data.shape[1] - 1)], verbose = 0)[1]
    
    return acc

def applyANN2(data, pts):
    dim = data.shape[1] - 1
    
    #Alle Ebene werden nacheinander verarbeiten
    model = tf.keras.Sequential()
    #Fügen input Ebenen hinzu
    model.add(tf.keras.Input(shape=((dim,))))
    #Dense Schicht wird konfiguriert mit 64 Neuronen 
    if (dim > 1):
        model.add(tf.keras.layers.Dense(dim, activation="relu"))
#    model.add(tf.keras.layers.Dense(1, activation="relu"))
#    model.add(tf.keras.layers.Dense(128, activation="relu"))
#    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    #Model erstellt und Optimierung wird konfiguriert
    model.compile(optimizer = "SGD", loss = "mse", metrics=['accuracy'])
    #Model wird auf meine Daten trainiert
    model.fit(data[:, :dim], data[:, dim], batch_size = 1, epochs = 10, verbose = 0)
    
    #Evaluiere ich gegen neue Daten
    acc = model.evaluate(pts[:, :(data.shape[1] - 1)], pts[:, (data.shape[1] - 1)], verbose = 0)[1]
    
    return acc
 
def eval_ml_methods(name, pattern_idx = 1, visualize = True):
    if (name == "ann"):
        data = st.gen_ann_prob_data(size = 5000, pattern_idx = pattern_idx)
    elif(name == "lda"):
        data = st.gen_lda_prob_data(size = 5000, pattern_idx = pattern_idx)
    elif(name == "knn"):
        data = st.gen_knn_prob_data(size = 5000, pattern_idx = pattern_idx)

    (data, pts_2) = normalizeData(data)
    
    if(name == "knn"):
        factor = 0.0
        #factor = 0.1
        #factor = 0.5
        #factor = 0.9
        
        data[:, 0] = np.power(data[:, 0], 2) * factor + data[:, 0] * (1 - factor)
        data[np.argwhere(data[:, 2] == 0), 0] += 0.05
        pts_2[:, 0] = np.power(pts_2[:, 0], 2) * factor + pts_2[:, 0] * (1 - factor)
        pts_2[np.argwhere(pts_2[:, 2] == 0), 0] += 0.05
        
        pts_2[:, 0] -= np.min(data[:, 0])
        data[:, 0] -= np.min(data[:, 0])
        pts_2[:, 0] /= np.max(data[:, 0])
        data[:, 0] /= np.max(data[:, 0])
        
        data[:, 0] *= 2
        pts_2[:, 0] *= 2
        data[:, 0] -= 1
        pts_2[:, 0] -= 1

        print(factor)
    
    res = applyML(data, pts_2)
    if (visualize):
        st.visualize(data, pts_2)
        plt.xlim((-1.1, 1.1))
        if(name == "knn"):
            plt.title(str(factor))
        plt.show(block = True)
    
    
    return res

def eval_pca(pattern_idx = 0, dim_cnt = 20):
    data = st.gen_pca_prob_data(5000, pattern_idx, dim_cnt)#1000 für KNN Unterschied
    (data, pts) = extract_pts(data)
    
    #===========================================================================
    # plt.scatter(data[np.argwhere(data[:, -1] == 0).squeeze(), 0], data[np.argwhere(data[:, -1] == 0).squeeze(), 2], color = "blue", alpha = 0.33)
    # plt.scatter(data[np.argwhere(data[:, -1] == 1).squeeze(), 0], data[np.argwhere(data[:, -1] == 1).squeeze(), 2], color = "red", alpha = 0.33)
    # plt.scatter(pts[:, 0], pts[:, 2], color = "green", alpha = 0.33)
    # plt.show(block = True)
    #===========================================================================
    
    (pca_data, pca_pts, var) = applyPCA(data, pts)
    pca_data = pca_data[:, [0, 1, 2, -1]]
    pca_pts = pca_pts[:, [0, 1, 2, -1]]
    
    lda_pred_pts = applyLDA(data, pts)
    acc = np.sum(lda_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tLDA %2.2f" % (acc, ))
    
    lda_pred_pts = applyLDA(pca_data, pca_pts)
    acc = np.sum(lda_pred_pts == pca_pts[:, (pca_data.shape[1] - 1)])/pca_pts.shape[0]
    print("\tLDA %2.2f" % (acc, ))
    
    knn_pred_pts = applyKNN(data, pts)
    acc = np.sum(knn_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tKNN %2.2f" % (acc, ))
    
    knn_pred_pts = applyKNN(pca_data, pca_pts)
    acc = np.sum(knn_pred_pts == pca_pts[:, (pca_data.shape[1] - 1)])/pca_pts.shape[0]
    print("\tKNN %2.2f" % (acc, ))

def score_ml_methods(name, pattern_idx, runs = 1000, overwrite = False):

    file_name = "res_" + name + "_" + str(pattern_idx) + ".npy"
    print(file_name)

    if (overwrite == True or os.path.exists(file_name) == False):
        res = list()
        for idx in range(runs):
            print("Processing run " + str(idx) + " of " + str(runs))
            res.append(eval_ml_methods(name, pattern_idx, False))
        res = np.asarray(res)
        np.save(file_name, res)
        
        print(np.mean(res, axis = 0))
        
    else:
        res = np.load(file_name)


def test():
    
    #Class 0 - jumping between 0 and 1
    #Class 1 - always the same thing (either 0 or 1)
    
    cnt = 8
    
    data = np.zeros((1000, 8))
    data = np.zeros((cnt, 2))
    
    
    class1_indices = np.random.permutation(range(0, int(cnt/2)))[:int(cnt/4)] 
    data[class1_indices, :] = 1
    
    class2_indices = np.random.permutation(range(int(cnt/2), cnt))
    for row in np.arange(1, data.shape[1] - 1 + 1, 2):
        data[class2_indices[:int(cnt/4)], row] = 1
    for row in np.arange(0, data.shape[1] - 1 + 1, 2):
        data[class2_indices[int(cnt/4):], row] = 1
    
    data = np.concatenate((data, np.expand_dims(np.concatenate((np.zeros(int(cnt/2)), np.ones(int(cnt/2))), axis = 0), axis = 1)), axis = 1)
    
    data = data[np.random.permutation(np.arange(cnt)), :]
    
    #data[:, :-1] += np.random.normal(0, 0.1, data[:, :-1].shape)
    
    pts = data[:100]
    data = data[100:]
    
    lda_pred_pts = applyLDA(data, pts)
    acc_lda = np.sum(lda_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tLDA %2.2f" % (acc_lda, ))
    
    knn_pred_pts = applyKNN(data, pts)
    acc_knn = np.sum(knn_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tKNN %2.2f" % (acc_knn, ))
    
    acc_ann = applyANN2(data, pts)
    print("\tANN %2.2f" % (acc_ann, ))
    
def test2():

    cnt = 200
    data = np.arange(0, cnt, 2)
    data2 = (np.arange(0, cnt, 2) + cnt/2) / 10
    
    data = np.concatenate((data, data2), axis = 0)
    data = np.expand_dims(data, axis = 1)

    data = np.concatenate((data, np.expand_dims(np.concatenate((np.zeros(int(cnt/2)), np.ones(int(cnt/2))), axis = 0), axis = 1)), axis = 1)
    
    data = data[np.random.permutation(np.arange(cnt)), :]
    
    #data[:, :-1] += np.random.normal(0, 0.1, data[:, :-1].shape)
    
    pts = data[:100]
    data = data[100:]
    
    lda_pred_pts = applyLDA(data, pts)
    acc_lda = np.sum(lda_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tLDA %2.2f" % (acc_lda, ))
    
    knn_pred_pts = applyKNN(data, pts)
    acc_knn = np.sum(knn_pred_pts == pts[:, (data.shape[1] - 1)])/pts.shape[0]
    print("\tKNN %2.2f" % (acc_knn, ))
    
    acc_ann = applyANN2(data, pts)
    print("\tANN %2.2f" % (acc_ann, ))

#===============================================================================
# test2()
# print("terminated")
# quit()
#===============================================================================

eval_ml_methods("ann", 1, True)# (pca und Ica noch nicht)
    
#score_ml_methods("lda", 2, 1000, True)# (pca und Ica noch nicht)
#score_ml_methods("knn", 1, 1000, True)# (pca und Ica noch nicht)

#eval_pca(0, 300)

print("terminate")










