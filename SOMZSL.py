# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:02:03 2023

@author: firat ismailoglu
If you use this code for research prurposes, please cite:
    "Zero-shot learning via self-organizing maps" by Firat Ismailoglu
    Neural Computing and Applications
    
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import io, spatial
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from pca_initialization import PCA_init #this must be in the folder where this script resides

"""
Required INPUTS
X_train ==>with size of:  number of training instances x dimension of feature (input) space 
X_test==>with size of:  number of test instances x dimension of feature space dimension

labels_train ==>is a column vector corresponding to labels of training instances
labels_test ==>is a column vector corresponding to labels of test instances

train_sig  ==>is a matrix, where each row corresponds to the semantic representation of a training (seen) class; thus its size is number of training classes x number of attributes (i.e., dimension of the semantic space)
test_sig  ==>is a matrix, where each row corresponds to the semantic representation of a test (unseen) class; thus its size is number of test classes x number of attributes (i.e., dimension of the semantic space)


"""



def normalizeFeature(x): #for feature normalization
	# x = N x d ( N:number of instances,d:feature dimension)
    x = x + 1e-10
    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x /np.reshape(feature_norm,(x.shape[0],1))
    return feat

########## Methods/Functions needed to build a SOM###########
def find_BMU(SOM,x): #returns the coordinates of winner cell of the SOM for x 
    distSq = (np.square(SOM - x)).sum(axis=2)
    coord= np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  
    return coord    

#weight update. here "step" controls the number of cells whose weight vector will be updated
def update_weights(SOM, train_ex, learn_rate, radius_sq, 
                   BMU_coord, reg_param,step=3): 
    g, h = BMU_coord
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)): #satir indisi
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = 2*np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * reg_param*(train_ex - SOM[i,j,:])   
    return SOM


##################### now SOMZSL############################
#Here we have two SOMs: SOM_in that corresponds to the SOM of the input space
# SOM_att that corresponds to the SOM of the attribute space    
    
### Phase-1 Updating the SOMs using the given training set
def train_SOMs(SOM_in,SOM_att, X_train,labels_train, train_sig, learn_rate = .1, radius_sq = 1, 
             lr_decay = .1, radius_decay = .1, epochs = 5,reg_param=0.01):
    
    learn_rate_0 = learn_rate #starting learning rate
    radius_0 = radius_sq
    n_train=X_train.shape[0] #number of training examples
    
    #here epochs is epochs:max_iter1
    for epoch in np.arange(0, epochs):     #for each epoch we visit all training examples
                                           #alternatively, for each epoch, one can get batch of training examples randomly
                                           #and can update the SOMs based on that batch
  
        for tr in range(n_train):
            coord_in= find_BMU(SOM_in, X_train[tr,:])
            coord_att = find_BMU(SOM_att,train_sig[labels_train[tr],:])
  
            SOM_in = update_weights(SOM_in, X_train[tr,:],  learn_rate, radius_sq, (coord_att[0],coord_att[1]),reg_param)
                                    #here is main trick of SOMZSL, we update SOM_in based on  the winner neuron of SOM_att
            
            SOM_att= update_weights(SOM_att, train_sig[labels_train[tr],:], 
                                 learn_rate, radius_sq, (coord_in[0],coord_in[1]),reg_param=1) 
              
        learn_rate = learn_rate_0 * np.exp(-(epoch / lr_decay))
        radius_sq = radius_0 * np.exp(-(epoch / lr_decay)) 
#here one can prefer different functions to decay the learning rate and the radius
    return SOM_in, SOM_att    

############# Phase 2 Updating the SOMs based on unlabeled test data
def test_SOMs(SOM_in,SOM_att, X_test, test_sig, learn_rate = .1, radius_sq = 1,  
             lr_decay = .1, radius_decay = .1, epochs = 10,reg_param1=0.01,reg_param2=5):    
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    n_test=X_test.shape[0]
    test_class_number=test_sig.shape[0]
        
    
    for epoch in np.arange(0, epochs):   #here epochs is max_iter2
              
        for te in range(n_test):
            coord_in = find_BMU(SOM_in, X_test[te,:])
            SOM_in = update_weights(SOM_in, X_test[te,:], 
                                   learn_rate, radius_sq, (coord_in[0],coord_in[1]),reg_param=reg_param1)
          
        for y in range(test_class_number):   
            coord_att = find_BMU(SOM_att, test_sig[y,:])
            SOM_att = update_weights(SOM_att, test_sig[y,:], 
                                 learn_rate, radius_sq, (coord_att[0],coord_att[1]),reg_param=reg_param2) #better choose relatively big reg_param2 such as 5
        # Update learning rate and radius                
        learn_rate = learn_rate_0 * np.exp(-epoch *lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * lr_decay) 

    return SOM_in, SOM_att

################## now we merge these methods to reach the final SOMZSL method
    
def SOMZSL(SOM_in,SOM_att, X_train,X_test,labels_train, train_sig, test_sig,maxiter1, maxiter2,learn_rate = 0.01, radius_sq = 3, 
         lr_decay = .1, radius_decay = .01, reg_param=0.1):

    
    [SOM_in_trained,SOM_att_trained]= train_SOMs(SOM_in,SOM_att, X_train,labels_train, train_sig, learn_rate = 0.01, radius_sq = 3, 
         lr_decay = .1, radius_decay = .01, epochs = maxiter1 ,reg_param=0.1)
    
    
    [SOM_in_full_trained, SOM_att_full_trained]=test_SOMs(SOM_in_trained,SOM_att_trained, X_test, test_sig, learn_rate = 0.01, radius_sq =3,  
         lr_decay = .01, radius_decay = .01, epochs = maxiter2,reg_param1=0.01,reg_param2=5)

    return SOM_in_full_trained, SOM_att_full_trained

#######  prediction with these SOMs###########   
def prediction(X_test,SOM_in,SOM_att,test_sig,labels_test):
    n_test=X_test.shape[0]
    predicts=np.zeros((n_test,1))

    for i in range(n_test):
        te=X_test[i,:]
        coord=find_BMU(SOM_in, te)
        #g_in, h_in = find_BMU(SOM_in, te)
        te_in_att=SOM_att[coord[0],coord[1],:]
        dists=(np.square(te_in_att-test_sig)).sum(axis=1)
        predicts[i,0]=np.argmin(dists)
    return  predicts

#########  Top-1 Accuracy Calculation###########
def top1_acc(true_test_labels,predicted_test_labels,test_sig):
   cm = confusion_matrix(true_test_labels, predicted_test_labels)
   cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   acc=sum(cm.diagonal())/test_sig.shape[0]
   return acc
############## Building intial SOMs##########
#For this we have three choices: full random, intialization with random instances, PCA initialization
################################################
def SOM_init(init_type,m,n,X_train,train_sig):
    [n_train,dim_in]=X_train.shape#dim_in equals to the dimension of the input space
    [n_tr_class,dim_att]=train_sig.shape ##dim_in equals to the dimension of the attribute space (i.e. number of attributes)
    SOM_in= np.random.random((m, n, dim_in))
    SOM_att= np.random.random((m, n, dim_att))
    if init_type=="full_random_init":       
        return SOM_in,SOM_att
    
    if init_type=="random_init_with_data":   
        for i in range(m):
            for j in range(n):
                SOM_in[i,j,:]=X_train[random.randint(0, n_train),:]
                SOM_att[i,j,:]=train_sig[random.randint(0, n_tr_class),:]
        return SOM_in,SOM_att
    
    if init_type=="pca_init":  
        SOM_in=PCA_init(SOM_in, X_train)
        SOM_att=PCA_init(SOM_att, train_sig)
        return SOM_in,SOM_att
#########################################################










