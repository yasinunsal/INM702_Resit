# INM702

for Task 1:
Model.py is the main file. All the model design and comparing models, plotting graphs are processed here. Model.py file calls methods in Task1_M1.py program which has forward and backward algorithms. Task1_M1.py includes Activitions.py file which has activation functions in it.

for Task 2:
Task2.ipynb will be used.


Coursework for INM702 Programming and Mathematics for AI - ANN Implementation


The task is about classification on the MNIST dataset. You can use other API’s/libraries for loading the dataset, but not for the neural network implementation. The point of this task is to develop a multi-layer neural network for classification using numpy. The task requires following sub-tasks: 

a. Implement sigmoid and ReLU layers For this sub-task, you should implement forward and backward pass for sigmoid and ReLU. You should consider presenting these activation functions in the report with any pros cons if they have. (10%) 

b. Implement softmax layer Implement softmax with both forward and backward pass. Present the softmax in the report along with any numerical issues when calculating the softmax function. (15%) 

c. Implement dropout Present dropout in the report. Implement inverted dropout. Forward and backward pass should be implemented. Note: Since the test performance is critical, it is also preferable to leaving the forward pass unchanged at test time. Therefore, in most implementations inverted dropout is employed to overcome the undesirable property of the original dropout. (15%) 

d. Implement a fully parametrizable neural network class You should implement a fully-connected NN class where with number of hidden layers, units, activation functions can be changed. In addition, you can add dropout or regularizer (L1 or L2). Report the parameters used (update rule, learning rate, decay, epochs, batch size) and include the plots in your report. (20%) 

e. Implement optimizer Implement any two optimizers of your choice. Briefly present the optimizers in the report. The optimizers can be flavours of gradient descent. For instance: Stochastic gradient descent (SGD) and SGD with momentum. SGD and mini-batch gradient descent, etc. (10%) 

f. Evaluate different neural network architectures/parameters, present and discuss your results. Be creative in the analysis and discussion. Evaluate different hyperparameters. For instance: different network architectures, activation functions, comparison of optimizers, L1/L2 performance comparison with dropout, etc. Support your results with plots/graph and discussion.
