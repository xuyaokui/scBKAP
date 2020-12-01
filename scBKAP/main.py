# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:57:02 2020

@author: hua'wei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import tensorflow as tf
from Autoencoder1 import Autoencoder
import csv
import phate
from bikmeans import biKmeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

def filte(data, args_name):
    file_path = args_name +".csv"
    a = data.shape[1]
    exist = (data > 0) * 1.0
    factor = np.ones(data.shape[1])
    res = ((np.dot(exist, factor))/a)*100  
    test = np.column_stack((res,data))
    with open(file_path, 'w', newline='') as fout:
        reader = test
        writer = csv.writer(fout, delimiter=',')
        for i in reader:
            if 5 < int(i[0]):
                writer.writerow(i)
    return

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

def autorunner(filter_data, epochs, h1, h2, args_name):

    batch_size = filter_data.shape[0]-1
    num = filter_data.shape[1]
    file_path = args_name +".csv"
    
    n_samples,_ = np.shape(filter_data)

    training_epochs = epochs
    display_step = 1

    autoencoder = Autoencoder(
        n_input = num,
        n_hidden1 = h1,
        n_hidden2 = h2,
        n_hidden3 = h1,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(filter_data, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))    

    print("Total cost: " + str(autoencoder.calc_total_cost(filter_data)))
    X_test_transform=autoencoder.transform(filter_data)
    X_test_reconstruct=autoencoder.reconstruct(filter_data)
    
    
    with open(file_path, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter=',')
        for i in X_test_reconstruct:
            writer.writerow(i)
            
    return X_test_reconstruct

def clust(data, label, pca_com, phate_com):
    pca = PCA(n_components=pca_com)
    b = pca.fit_transform(data)
    phate_op = phate.PHATE(n_components=phate_com)
    data_phate = phate_op.fit_transform(b)
    
    c = label.max()
    centList,clusterAssment = biKmeans(data_phate,c)
    julei = clusterAssment[:,0]
    y=np.array(julei)
    julei = y.ravel()
    
    print('NMI value is %f \n' % nmi(julei.flatten(),label.flatten()))
    print('ARI value is %f \n' % ari(julei.flatten(),label.flatten()))
    print('HOM value is %f \n' % metrics.homogeneity_score(julei,label))
    print('AMI value is %f \n' % metrics.adjusted_mutual_info_score(label, julei))
    
    return julei
