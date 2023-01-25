# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:06:30 2022

@author: FIsmailoglu
PCA based initialization ofr SOM
"""

import numpy as np

from sklearn.decomposition import PCA

def PCA_init(som, data):

    som_copy=np.copy(som)
    cols = som_copy.shape[1]
    nnodes=som_copy.shape[0]*som_copy.shape[1]

    coord = np.zeros((nnodes, 2)) #
    pca_components = 2

    for i in range(0, nnodes):
        coord[i, 0] = int(i / cols)  # x
        coord[i, 1] = int(i % cols)  # y
        
    mx = np.max(coord, axis=0)
    mn = np.min(coord, axis=0)
    coord = (coord - mn)/(mx-mn) #normalizing the coordinates
    coord = (coord - .5)*2
    me = np.mean(data, 0)
    data = (data - me) #datadaki her kolonun merkezi 0'a cekiliyor.
    tmp_matrix = np.tile(me, (nnodes, 1))#toplam node saysisi kadar nodedan oluşan ve her satiri datanin ana mean vektörü olan matris 

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components) # RandomizedPCA is deprecated.
    pca = PCA(n_components=pca_components, svd_solver='randomized')
    pca.fit(data)#her kolonun ortalamasi 0 olmuş data
    eigvec = pca.components_ #aradigimiz ilk iki eig vektörlerin matrisi
    eigval = pca.explained_variance_
    norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
    eigvec = ((eigvec.T/norms)*eigval).T

    for j in range(nnodes):
        for i in range(eigvec.shape[0]):
            tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

    init_matrix = np.around(tmp_matrix, decimals=6)
    coord_matrix = np.zeros((nnodes, 2)) #her brid node'un coordinatlarini tutuyor

    for i in range(0, nnodes):
        coord_matrix[i, 0] = int(i / cols)  # x
        coord_matrix[i, 1] = int(i % cols)  # y
        
    for i in range(cols):
        for j in range(cols):
            som_copy[i,j,:]=init_matrix[i*cols+j,:]
    
    return som_copy
            
