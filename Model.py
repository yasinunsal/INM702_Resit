#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import Task1_M1


Y_label=10

train_data = pd.read_csv("./MNIST_data/mnist_train.csv")
test_data= pd.read_csv("./MNIST_data/mnist_test.csv")

"""
___________________________________________________________
Prepare Train data
___________________________________________________________
""" 
# separate labels and pixels
train_labels=np.array(train_data.loc[:,'label'])
train_data=np.array(train_data.loc[:,train_data.columns!='label'])

train_data_row = train_data.shape[0]
train_data_column = train_data.shape[1]

# Reshape 
train_data=np.reshape(train_data,[train_data_column ,train_data_row])
train_label=np.zeros((Y_label,train_data_row))

# One-Hot encoding for train labels
for col in range (train_data.shape[0]):
    val=train_labels[col]
    for row in range (Y_label):
        if (val==row):
            train_label[val,col]=1

print("")          
print("train_data shape="+str(np.shape(train_data)))
print("train_label shape="+str(np.shape(train_label)))


"""
___________________________________________________________
Prepare Test data
___________________________________________________________
""" 
# separate labels and pixels
test_labels=np.array(test_data.loc[:,'label'])
test_data=np.array(test_data.loc[:,test_data.columns!='label'])

test_data_row = test_data.shape[0]
test_data_column = test_data.shape[1]

# Reshape 
test_data=np.reshape(test_data,[test_data_column ,test_data_row])
test_label=np.zeros((Y_label,test_data_row))

# One-Hot encoding for test labels
for col in range (test_data.shape[0]):
    val=test_labels[col]
    for row in range (Y_label):
        if (val==row):
            test_label[val,col]=1   
print("")
print("test_data shape="+str(np.shape(test_data)))
print("test_label shape="+str(np.shape(test_label)))

"""
___________________________________________________________
Test Data Porperties
___________________________________________________________
""" 

y_value=np.zeros((1,10))
for i in range (10):
    print("Number of ",i,"=",np.count_nonzero(test_labels==i))
    y_value[0,i-1]= np.count_nonzero(test_labels==i)

"""
___________________________________________________________
MODEL COMPARING
___________________________________________________________
""" 
print("")
print("Model Comparison")

accuracy_list=np.zeros(4)

def cost_graph(cost,color):
    x_value=list(range(1,len(cost)+1))
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(x_value,cost,0.,color=color)

def perform_model(model, layers_dims, AF1, AF2, learning_rate, num_iterations,color):
    
    print("")
    print("_________________________________")
    print("Model No:",model)
    print("Dimensions:",layers_dims)
    print("Activation Functions (Hidden layers,Output layer):",AF1, AF2)
    print("Learning rate:",learning_rate)
    print("Number of iterations:",num_iterations)
    print("Colour:",color)    
    print("")
    
    print("Training is starting..")
    temp_train_data = copy.deepcopy(train_data)
    temp_train_label = copy.deepcopy(train_label)
    
    parameters, cost = Task1_M1.L_layer_model(temp_train_data, temp_train_label, layers_dims, AF1, AF2, learning_rate = learning_rate, num_iterations=num_iterations ) 
    print("Training finished..")
    
    temp_test_data = copy.deepcopy(test_data)
    temp_test_label = copy.deepcopy(test_label)
    print("Predicting test data..")
    accuracy_list[model-1], y_pred = Task1_M1.predict(temp_test_data, temp_test_label, layers_dims, parameters, AF1, AF2 )

    print("Accuracy is: ",accuracy_list[model-1])
    
    cost_graph(cost,color)
    
     

"""
___________________________________________________________
MODEL 1
___________________________________________________________
""" 

layers = [train_data.shape[0],500,300,100,train_label.shape[0]]
hidden_activation = "relu"
output_activation = "softmax"
lr = 0.75
iterations = 30
color="r"
model_no=1

perform_model(model_no, layers, hidden_activation, output_activation, lr, iterations, color)

"""
___________________________________________________________
MODEL 2
___________________________________________________________
""" 

layers = [train_data.shape[0],700,train_label.shape[0]]
hidden_activation = "relu"
output_activation = "sigmoid"
lr = 0.001
iterations = 30
color="b"
model_no=2


perform_model(model_no, layers, hidden_activation, output_activation, lr, iterations, color)

"""
___________________________________________________________
MODEL 3
___________________________________________________________
""" 

layers = [train_data.shape[0],700,300, 40, train_label.shape[0]]
hidden_activation = "sigmoid"
output_activation = "softmax"
lr = 0.15
iterations = 30
color="g"
model_no=3

perform_model(model_no, layers, hidden_activation, output_activation, lr, iterations, color)

"""
___________________________________________________________
MODEL 4
___________________________________________________________
""" 
layers = [train_data.shape[0],900,train_label.shape[0]]
hidden_activation = "sigmoid"
output_activation = "softmax"
lr = 0.05
iterations = 30
color="y"
model_no=4

perform_model(model_no, layers, hidden_activation, output_activation, lr, iterations, color)



print("")
print("_________________________________")
print("Accuracy",accuracy_list)

fig, ax = plt.subplots()
bar_width = 0.35
X = np.arange(4)

p1 = plt.bar(X, accuracy_list, bar_width, color='b',
label='Accuracy')
plt.xlabel('Accuracy of Each Model')
plt.ylabel('Accuracy')
plt.xticks(X + (bar_width/2) , ("Model 1", "Model 2", 
"Model 3", "Model 4"))
plt.legend()

plt.tight_layout()
plt.show()