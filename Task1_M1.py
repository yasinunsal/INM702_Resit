#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:51:33 2023

@author: berrinulus

1- Parameter Initialization
2- Optimization
    2a- Forward Propagation
    2b- Forward Propagation with Activation Function
    2c- Forward Propagation Model 
    
    2d- Cost Computation
    2e- Backward Propagation
    2f- Backward Propagation with Activation Function
    2g- Backward Propagation Model 
    2h- Update Parameters
3- Multi Layer Model     
"""

#___________________________________________________
import numpy as np
import Activations

#___________________________________________________
# 1- Parameter Initialization
#___________________________________________________
def initialize_parameters(layer_dims):
    """
    Arguments: 
        layer_dims : python array (list) containing dimension of each layer in our network
            length of the list give layer count, list values gives us nodes that every layer will have
    Returns:
        parameters : python dictionary containing weights and bias "W1", "b1", "WL", "bL" 
            wl.shape : (layer_dims[l], layer_dims[l-1])
            bl.shape : (layer_dims[l], 1)
    """   
    np.random.seed(8)
    parameters = {}
    L = len(layer_dims)            
    
    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 #/ np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters
  
    
#___________________________________________________
# 2- Optimization
#___________________________________________________


#___________________________________________________
# 2a- Forward Propagation
#___________________________________________________
def linear_forward(A, W, b):
    """
    Arguments: 
        A : Activations from previous layer, or input data for the first iteration
            A.shape = (size of previous layer, number of examples)
        W : Weight matrix
            W.shape = (size of current layer, size of previous layer)
        b : bias vector 
            b.shape = (size of current layer, 1)
            
    Returns:
        Z: input of the activation function - pre-activation values
        cache : tuple that contains "A", "W", "b" matrices to compute backward propagation
    """     
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache

#___________________________________________________
# 2b- Forward Propagation with Activation Function
#___________________________________________________
def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments: 
        A_prev : Activations from previous layer, or input data for the first iteration
            A.shape = (size of previous layer, number of examples)
        W : Weight matrix
            W.shape = (size of current layer, size of previous layer)
        b : bias vector 
            b.shape = (size of current layer, 1)
        activation : activation function name to be used in this layer
            
    Returns:
        A: Output of the activation function - post-activation values
        cache : tuple that contains "linear_cache" and "activation_cache" to compute backward propagation
    """ 
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Activations.sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Activations.relu(Z) 
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Activations.softmax(Z)
    elif activation == "tan":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Activations.tanh(Z)
   
    cache = (linear_cache, activation_cache)
    return A, cache

        
#___________________________________________________
# 2c- Forward Propagation Model
#___________________________________________________
def l_model_forward(X, parameters, AF1, AF2):
    """
    Forward Propagation for [LINEAR->ACTIVATION_FUNCTION_1]*(L-1) -> [LINEAR->ACTIVATION_FUNCTION_2) 
    Arguments: 
        X : input data
            X.shape = (input size, number of examples)
        parameters : python dictionary containing weights and bias "W1", "b1", "WL", "bL"

            W : Weight matrix
                W.shape = (size of current layer, size of previous layer)
            b : bias vector 
                b.shape = (size of current layer, 1)
        
        AF1 : activation function name to be used in the hidden layers
        AF2 : activation function name to be used in the output layer
            
    Returns:
        AL: Activation value from the output layer
        cache : array that contains cache of linear_activation_forward function that
            contains "linear_cache" and "activation_cache" of all layers
            There are L of cache indexed from 0 to (L-1)
    """     
    
    caches = []
    A = X
    L = len(parameters) // 2   
    

    # [LINEAR->ACTIVATION_FUNCTION_1]*(L-1)
    # loop will start at index 1, because index 0 is Input layer X              
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = AF1)
        caches.append(cache)
        
        
    # [LINEAR->ACTIVATION_FUNCTION_2] for the output layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = AF2)
    
    caches.append(cache)               
    return AL, caches    

#___________________________________________________
# 2d- Cost Computation
#___________________________________________________
def compute_cost(AL, Y,  epsilon=1e-15):
    """
    Forward Propagation for [LINEAR->ACTIVATION_FUNCTION_1]*(L-1) -> [LINEAR->ACTIVATION_FUNCTION_2) 
    Arguments: 
        AL: Probability Vector - Activation value from the output layer
            AL.shape = (10, number of examples)
        Y : True label vector
            Y.shape = (10, number of examples)       
        
    Returns:
        Cost : Cross Entrpoy cost
            J = - 1/m * ( Σ  ( y * log(AL) + (1-y) * log(1-AL) ) )
    """  
    
    m = Y.shape[1] 
    #cost = (-1 / m) * np.sum(   np.multiply(Y, np.log(AL + 1e-15)) + np.multiply( (1 - Y), np.log(1 - AL + 1e-15) )     )
    
    cost = (-1 / m) * np.sum(   np.multiply( np.log(AL + 1e-15),Y ) + np.multiply( (1 - Y), np.log(1 - AL + 1e-15) )     )
    
    return cost

def calculate_mse(AL, Y):
    return (Y - AL) ** 2

def cross_entropy_loss(AL, Y, epsilon=1e-15):
    predictions = np.clip(AL, epsilon, 1 - epsilon)  
    ce_loss = -np.sum(Y * np.log(predictions + epsilon)) / len(predictions)
    return ce_loss

#___________________________________________________
# 2e- Backward Propagation
#___________________________________________________
def linear_backward(dZ, cache):
    """
    Linear portion of Backward Propagation for layer l
    
    Arguments: 
        dZ: Gradient of the cost respect to linear output of current layer l
        cache : Tuple of values ( A_prev, W, b) that comes from forward propagation in current layer l
        
    Returns:
        dA_prev : Gradient of the cost respect to the activation of previous layer (l-1)
            dA_prev.shape = (size of previous layer, number of examples)
        dW :  Gradient of the cost respect to W
            dW.shape = (size of current layer, size of previous layer)
        db : Gradient of the cost respect to b
            db.shape = (size of current layer, 1)
        
    """  
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = ( 1 / m ) * np.dot(dZ, A_prev.T)  
    db = ( 1 / m ) * np.sum(dZ, axis=1, keepdims=True);
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

#__________________________________________________
# 2f- Backward Propagation with Activation Function
#___________________________________________________
def linear_activation_backward(dA, cache, activation):
    """
    Backward Propagation for LINEAR->ACTIVATION
    
    Arguments: 
        dA : Post activation gradient for current layer l        
        cache : Tuple of values ( linear_cache, activation_cache) that stored for backward propagation
        activation : activation function name to be used in this layer
        
    Returns:
        dA_prev : Gradient of the cost respect to the activation of previous layer (l-1)
            dA_prev.shape = (size of previous layer, number of examples)
        dW :  Gradient of the cost respect to W
            dW.shape = (size of current layer, size of previous layer)
        db : Gradient of the cost respect to b
            db.shape = (size of current layer, 1)
    """   
    
    linear_cache, activation_cache = cache
    
    if activation == 'sigmoid':
        dZ = Activations.sigmoid_derivative(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = Activations.relu_derivative(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'softmax':
        dZ = Activations.softmax_derivative(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "tan":
        dZ = Activations.tanh_derivative(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    
    
    return dA_prev, dW, db
    
#__________________________________________________
# 2g- Backward Propagation Model 
#___________________________________________________  
def l_model_backward(AL, Y, caches, AF1, AF2):
    """
    Backward Propagation 
    [Linear->ACTIVATION_FUNCTION_2] * (L-1) -> [Linear->ACTIVATION_FUNCTION_1]
    
    Arguments: 
        AL: Activation values - Probability vector from the output layer 
        Y : True label vector
        caches : list of caches cantaining:
            every cache of linear_activation_forward with ACTIVATION_FUNCTION_1
            cache of linear_activation_forward with ACTIVATION_FUNCTION_2           
        AF1 : activation function name to be used in the hidden layers
        AF2 : activation function name to be used in the output layer
        
    Returns:
        grads : A dictionary with gradients
            grads["dA"+str(l)] = ...
            grads["dW"+str(l)] = ...
            grads["db"+str(l)] = ...
    """ 
    
    
    grads = {}
    L = len(caches) # the number of layers
    dAL = - (np.divide(Y, (AL + 1e-15))  - np.divide(1 - Y, (1 - AL + 1e-15 )))
    
    current_cache = caches[L-1]
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, activation = AF2)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = AF1 )
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads
     

#__________________________________________________
# 2h- Update Parameters
#___________________________________________________ 
def update_parameters(parameters, grads, learning_rate):
    """
    Arguments: 
        params: dictionary containin parameters
            params["W"+str(l)]
            params["b"+str(l)]
        grads : dictionary with gradients, output of L_model_backward
            grads["dA"+str(l)]
            grads["dW"+str(l)] 
            grads["db"+str(l)]
    Returns:
        params: dictionary containin updated parameters
            params["W"+str(l)]
            params["b"+str(l)]
    """   
    
    # number of layers
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db" + str(l+1)])
    return parameters
    

#__________________________________________________
# 3- Multi Layer Model
#___________________________________________________ 
def L_layer_model(X, Y, layers_dims, AF1, AF2, learning_rate , num_iterations, print_cost=True ):

    cost_plot=np.zeros(num_iterations)
    parameters = initialize_parameters(layers_dims)
     
    for i in range(0, num_iterations):
        AL, caches = l_model_forward(X, parameters, AF1, AF2)
        cost = compute_cost(AL, Y)
        #cost = cross_entropy_loss(AL, Y)
        
        if print_cost and i % 10 == 0 and i!= 0:
            print("Epoch: ", i, "Cost: ", cost)
        grads = l_model_backward(AL, Y, caches, AF1, AF2)
        parameters = update_parameters(parameters, grads, learning_rate) 
        cost_plot[i]=cost;
        
    return parameters, cost_plot

#__________________________________________________
# 4-Prediction
#___________________________________________________ 
def predict(X, Y, layers_dims, parameters,  AF1, AF2):
    
    y_pred_temp, caches = l_model_forward(X, parameters, AF1, AF2)
    
    y_pred = np.argmax(y_pred_temp, axis=0)
    y_true = np.argmax(Y, axis=0)

    
    correct_prediction = 0
    for i in range(len(y_pred)):
        if (y_pred[i]==y_true[i]):
            correct_prediction = correct_prediction + 1
        
    score = correct_prediction / len(y_pred)
    
    return score, y_pred
    
    
    
 
    