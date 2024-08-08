# Authors 
# Berrin Ulus
# Yasin Unsal

import numpy as np
import math


"""
____________________________________________________________________________
Calculates the sigmoid for each row of the input x. Function is:
    sigmoid(x) = 1/ 1+e^(-x)
This function works for a row vector and also for matrices of shape (m,n).
    
Argument:
    x : numpy matrix of shape (m,n)

    Returns:
    r_sigmoid : A numpy matrix equal to the sigmoid of x, of shape (m,n)
____________________________________________________________________________
"""
def sigmoid(Z):
      
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

"""
____________________________________________________________________________
Calculates the derivative of sigmoid for each row of the input x. Function is:
    derivative_sigmoid(x) = x*(1-x)

Argument:
    x : numpy matrix of shape (m,n)

    Returns:
    ds : A numpy matrix equal to the sigmoid of x, of shape (m,n)
____________________________________________________________________________
"""
def sigmoid_derivative(dA, cache):  
    
    sigmo, cache_temp = sigmoid(dA)
    dZ = np.multiply(sigmo, ( 1 - sigmo ))
    return dZ


"""
____________________________________________________________________________
Calculates the relu for each row of the input x.
This function works for a row vector and also for matrices of shape (m,n).
    
Argument:
    x : numpy matrix of shape (m,n)

    Returns:
    r_relu : A numpy matrix equal to the relu of x, of shape (m,n)
____________________________________________________________________________
"""
def relu(Z):
    A = np.maximum(0,Z)    
    cache = Z 
    return A, cache
"""
____________________________________________________________________________
Calculates the derivative of sigmoid for each row of the input x. Function is:
    derivative_relu(x) = 1 for x>0, 0 for x<=0
____________________________________________________________________________
"""
def relu_derivative(dA, cache):
    
    
    dZ = (dA > 0) * 1
    return dZ
    
"""
____________________________________________________________________________
Calculates the softmax for each row of the input x.
    softmax(X) = e^Xi /( Σ e^Xi) 
____________________________________________________________________________
"""
def softmax(Z):
    
    
# Apply exp() element-wise to x
    e_x = np.exp(Z)
# Divide to vector that sums each row of x_exp  
    A = e_x / np.sum( e_x + 1e-15)  
    
    activation_cache=Z
    return A,activation_cache   

"""
____________________________________________________________________________
Calculates the derivative of sigmoid for each row of the input Z. Function is:
    derivative_softmax(x) = for i=j => s(i) * (1-s(i))
                            for i!=j => -s[i] * s[j]
____________________________________________________________________________
"""
def softmax_derivative(Z,cache):
    
    length = Z.shape[0]
    width = Z.shape[1]
    
    Z=cache
    dZ=np.zeros((width,length))
    Z=np.transpose(Z)
    for row in range (0,width):
            den=(np.sum(np.exp(Z[row,:])))*(np.sum(np.exp(Z[row,:])))
            if den == 0:
                den += 1e-6
            for col in range (0,length):
                sums=0
                for j in range (0,length):
                    if (j!=col):
                        sums=sums+(math.exp(Z[row,j]))
                
                dZ[row,col]=(math.exp(Z[row,col])*sums)/den          
    dZ=np.transpose(dZ)
    Z=np.transpose(Z)

    assert (dZ.shape == Z.shape)
    return dZ
    
"""
____________________________________________________________________________
Calculates tan for each row of the input x.
This function works for a row vector and also for matrices of shape (m,n).
    
Argument:
    x : numpy matrix of shape (m,n)

    Returns:
    r_tanx : A numpy matrix equal to the relu of x, of shape (m,n)
____________________________________________________________________________
"""
def tanh(x):
    cache = x
    r_tanx = np.tanh(x)
    return r_tanx, cache;
"""
____________________________________________________________________________
Calculates derivative of tax(x) for each row of the input x.
This function works for a row vector and also for matrices of shape (m,n).
    
Argument:
    x : numpy matrix of shape (m,n)

    Returns:
    dZ : A numpy matrix equal to the relu of x, of shape (m,n)
____________________________________________________________________________
"""
def tanh_derivative(x, cache):
    dZ = 1 - np.tanh(x)**2;
    return dZ

