#!/usr/bin/env python
# coding: utf-8

# ## This is the TRIPODS2022 Derivatives group's k-derivative code. 
# Given a function, this code takes k-number(s) of discrete derivative as regressors in order to predict the next numbers in that function's sequence.  Future experimentation will include testing a variety of functions, the number of predicted values, etc. to study how it affects predictive accuracy/variance and from that, find an ideal number of regressors to use that improves the performance of neural network prediction models. Additionally, study how differing function affects accuracy and define what makes a function "complicated." 

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
import math
from math import log
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import time


# In[ ]:


# Takes another derivative for use in kthderiv below (list size decreases by 1)
def derivative_new(my_list):
    '''
    Parameters:
        my_list (list): list to take derivatives of
    
    Returns:
        list containing differences of consecutive terms of my_list
    '''
    
    return [my_list[i]-my_list[i-1] for i in range(1,len(my_list))]

def kthderiv(extended_list, k, original_length):
    '''
    Parameters:
        extended_list (list): list to take derivatives of plus at least k predictions
        k (int): number of derivatives
        original_length (int): length of original list
    
    Returns:
        derivatives (numpy array): the ith column has the ith derivative of the list, up to the kth derivative
    '''
    
    derivatives = extended_list[:original_length]
    for i in range(1,k+1):
        next_deriv = extended_list[:original_length + i]
        for j in range(i):
            next_deriv = derivative_new(next_deriv)
        derivatives = np.column_stack((derivatives, next_deriv))
    return derivatives


# In[ ]:


# Creates a neural network with three hidden layers
def forecast(x, y, x_pred, dim, layers, neurons):
    '''
    Trains a model on x and y and returns the array of predictions for x_pred
    
    Parameters:
        x (numpy array): each row is a single input, contains a time column and the appropriate derivatives
        y (numpy array): k x 1 array of all of the training labels
        x_pred (numpy array): array with row length the same as x, used as input for predictions
        dim (int): number of columns in x
        layers (int): number of layers (excluding output)
        neurons (int): total number of neurons
    
    Returns:
        pred: predicted values for x-coordinates in x_pred
    '''
    
    model = Sequential()
    layer_size = 2**(layers - 1) * neurons//(2**(layers) - 1)
    model.add(Dense(layer_size, input_dim=dim, activation='relu'))
    
    for i in range(layers - 1):
        layer_size = layer_size//2
        model.add(Dense(layer_size, activation='relu')) # Add hidden layer half the size of the previous one
        
    model.add(Dense(1)) # Output
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x,y,verbose=0,epochs=512)
    pred = model.predict(x_pred) # generate predicted values (output)
    return pred

def kderiv_predict_nonrecursive_new(x, y, x_pred, k_derivs, pred_length, layers, neurons): 
    '''
    Parameters:
        x (list): x-coordinates of known values
        y (list): known function values
        x_pred (list): original x-coordinates plus x-coordinates we want predictions for
        k_derivs (int): number of derivatives to use
        pred_length (int): number of predictions
        layers (int): number of layers (excluding output)
        neurons (int): total number of neurons
        
    Returns:
        updated_pred: new predictions based on derivative regressors
    '''
    
    x_pred_extended = x_pred + [x_pred[-1] + 1 + i for i in range(k_derivs)] # extend x_pred by k_derivs to compensate for decrease in length from differentiating
    pred_extended = forecast(x, y, x_pred_extended, 1, layers, neurons) # forecast function values for x-coordinates in x_pred_extended
    pred_extended = list(np.squeeze(pred_extended)) # turn pred_extended into a list of floats
    x_derivs = kthderiv(y + pred_extended[len(y):len(y) + k_derivs], k_derivs, len(y)) # create array of derivatives regressors
    x_pred_multi = kthderiv(y + pred_extended[len(y):len(y) + pred_length + k_derivs], k_derivs, len(y) + pred_length) # create array to be fed into the network
    if k_derivs > 0:
        y = np.array(y)
    updated_pred = forecast(x_derivs, y, x_pred_multi, k_derivs+1, layers, neurons) # forecast based on added regressors
    return updated_pred # returns final predicted values


# In[ ]:


# Making predictions based on a x-sized set, function in y, for a certain amount of real numbers
x = [a for a in range(100)] # Edit range to vary size of training set 
y = [a**2 + math.sin(a)/10 for a in range(100)] # Edit function, number of predicted numbers
x_pred = [a for a in range(110)] 
pred = kderiv_predict_nonrecursive_new(x, y, x_pred, 3, 10, 3, 1750)
y_pred = np.array([a**2 + math.sin(a)/10 for a in range(110)])
y = np.array(y)

#Root-mean-square deviation (RMSD) calculates the difference between our values predicted and the true values
score = np.sqrt(metrics.mean_squared_error(pred,y_pred))
print(f"Final score (RMSE): {score}") 
# Caluclates the percent variance: increase or decrease in an account over time as a percentage of the total account value
print(((y.std()-score)/y.std())*100, "percent of variance explained") # Higher variance (60-75%+) indicates stronger strength of association


# In[ ]:


# Example to generate results for polynomials

polynomial_results = {}
for i in range(1,11):
    for deriv in range(5):
        for pred_length in range(10, 100, 10):
            x = [a for a in range(100)]
            y = [a**i for a in range(100)]
            x_pred = [a for a in range(110)] 
            pred = kderiv_predict_nonrecursive_new(x, y, x_pred, deriv, pred_length, 3, 1750)
            y_pred = np.array([a**i for a in range(110)])
            y = np.array(y)
            score = np.sqrt(metrics.mean_squared_error(pred,y_pred))
            print(f"Final score (RMSE): {score}") 
            variance_explained = ((y.std()-score)/y.std())*100
            print(variance_explained, "percent of variance explained")
            polynomial_results[i, deriv, pred_length] = variance_explained

