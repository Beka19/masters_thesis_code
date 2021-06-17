'''
Created on 30.07.2020

@author: Beka
'''
import matplotlib

import matplotlib.pyplot as plt

import numpy as np
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.neighbors

size = 500

mean0 = 150
sigma0 = 50
data0 = 1/(sigma0 * np.sqrt(2 * np.pi)) * np.exp(-1/2*np.power(((np.arange(size) - mean0)/sigma0), 2))
data0 -= 0.0035

c_flags = np.zeros(data0.shape)
c_flags[np.argwhere(data0 > 0).squeeze()] = 1

x = np.arange(0, 50, 0.1)
noise = 2/np.pi*np.arcsin(np.sin(x))*50

data0 = np.expand_dims(data0, axis = 0)
noise = np.expand_dims(noise, axis = 0)

data = np.concatenate((data0, noise), axis = 0)

mixed_data = np.zeros((2, size))
weights = [[0.4, 0.6], [0.6, 0.4]]
for idx_0 in range(2):
    for idx_1 in range(2):
        mixed_data[idx_0, :] += data[idx_1, :] * weights[idx_0][idx_1]

ica = sklearn.decomposition.FastICA()
ica_data = ica.fit_transform(mixed_data.transpose()).transpose()

(fig, axes) = plt.subplots(3, 1)

mask0 = np.argwhere(c_flags == 0).squeeze()
mask1 = np.argwhere(c_flags == 1).squeeze()
marker_size = 3

pts_idx = np.random.permutation(np.arange(0, size, 1))[:50]

axes[0].scatter(np.arange(0, size)[mask0], data[0, mask0], color = "blue", s = marker_size)
axes[0].scatter(np.arange(0, size)[mask1], data[0, mask1], color = "red", s = marker_size)
axes[0].scatter(np.arange(0, size)[pts_idx], data[0, pts_idx], color = "green", s = marker_size)
axes[0].twinx().scatter(np.arange(0, size), data[1, :], color = "black", s = marker_size)

axes[1].scatter(np.arange(0, size)[mask0], mixed_data[0, mask0], color = "blue", s = marker_size)
axes[1].scatter(np.arange(0, size)[mask1], mixed_data[0, mask1], color = "red", s = marker_size)
axes[1].scatter(np.arange(0, size)[pts_idx], mixed_data[0, pts_idx], color = "green", s = marker_size)
axes[1].scatter(np.arange(0, size), mixed_data[1, :], color = "black", s = marker_size)

axes[2].scatter(np.arange(0, size)[mask0], ica_data[0, mask0], color = "blue", s = marker_size)
axes[2].scatter(np.arange(0, size)[mask1], ica_data[0, mask1], color = "red", s = marker_size)
axes[2].scatter(np.arange(0, size)[pts_idx], ica_data[0, pts_idx], color = "green", s = marker_size)
axes[2].scatter(np.arange(0, size), ica_data[1, :], color = "black", s = marker_size)

train_flags = np.delete(c_flags, pts_idx)
pts_flags = c_flags[pts_idx]

train_data_mixed = np.delete(mixed_data, pts_idx, axis = 1)
pts_mixed = mixed_data[:, pts_idx]

clf = sklearn.neighbors.KNeighborsClassifier(5)
#clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()# sklearn.neighbors.KNeighborsClassifier(5)
clf.fit(train_data_mixed.transpose(), train_flags)
clf_pts_flags = clf.predict(pts_mixed.transpose())
print("clf accuracy: %1.2f" % (np.sum(clf_pts_flags == pts_flags)/len(pts_mixed[0]), ))
 
train_data_ica = np.delete(ica_data, pts_idx, axis = 1)
pts_ica = ica_data[:, pts_idx]

clf = sklearn.neighbors.KNeighborsClassifier(5)

clf.fit(train_data_ica.transpose(), train_flags)
clf_pts_flags = clf.predict(pts_ica.transpose())
print("clf accuracy: %1.2f" % (np.sum(clf_pts_flags == pts_flags)/len(pts_ica[0]), ))

plt.show(block = True)

