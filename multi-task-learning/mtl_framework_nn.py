import numpy as np
import random
import math
import matplotlib.pyplot as plt
from functions import *


"""Markdown

"""

class MTL_ADD_Network_Layer:
    """
    This class defines an individual fully connected layer's connection to another layer for the additive MTL network. 
        In this implementation, MTL_ADD_Network_Layer defines the connection between two layers (an input layer and an 
        output layer). We also utilize matrix calculations to perform forward and backward propagations. You'll end up 
        keeping a list of network layers which represents the full network. The output layer in one MTL_ADD_Network_Layer 
        becomes the input layer in the next MTL_ADD_Network_Layer.
    """
    def __init__(self, input_size, output_size, n_tasks, activation=sigmoid, activation_prime=sigmoid_prime):
        """
        input_size - This is a scalar value representing the number of input neurons to include
        output_size - This is a scalar value representing the number of output neurons to include
        n_tasks - This is a scalar value representing the number of tasks trained simultaneously in the network
        activation - This is a function representing what function to use to activate the linear combination of the 
                        layers connections and the input layers input
        activation_prime - This is a function representing what function derivative to use in backpropagation

        This function initializes a MTL_ADD_Network_Layer by providing the number of input neurons and number of output neurons. 
            You can also provide the activation function and its derivative, but the default is sigmoid
        """
        self.weights = np.random.normal(0, 1, size=(input_size, output_size))
        #m_weights is a list of weight matrices that represent the task-specific masked weights
        self.m_weights = [np.random.normal(0, 1, size=(input_size, output_size)) for t in range(n_tasks)]
        self.bias = np.random.normal(0, 1, size=(output_size,))
        #m_bias is a list of bias vectors that represent the task-specific masked bias
        self.m_bias = [np.random.normal(0, 1, size=(output_size,)) for t in range(n_tasks)]

        self.activation = activation
        self.activation_prime = activation_prime
        
        self.input = None
        self.output = None
        
    def forward_propagation(self, input_data, task):
        """
        input_data - This should be a scalar vector of size (input_size,)
        task - This should be a scalar representing the task of the observation being currently trained in the network
        This function performs the forward propagation pass for the network layer
        """
        self.input = input_data
        combined_weights = self.weights + self.m_weights[task]
        combined_bias = self.bias + self.m_bias[task]
        self.output = self.activation(matmul(combined_weights.T,self.input)+combined_bias)
        return self.output
        
    def backward_propagation(self, output_error, shared_learning_rate, task_learning_rate, task):
        """
        output_error - This should be a scalar vector of size (output_size,) which equals dE/dy of a given NetworkLayer
        learning_rate - This should be a scalar value representing how much you want to update the weights and biases
                            on each example
        task - This should be a scalar representing the task of the observation being currently trained in the network
        This function performs the backward propagation pass for the network layer. It takes it's MTL_ADD_Network_Layer 
            output_error (dE/dy) as input, and calculates dE/dmW, dE/dmB, dE/dW, dE/dB to update the MTL_ADD_Network_Layer's 
            parameters, and it calculates dE/dx which becomes dE/dy for the next MTL_ADD_Network_Layer visited in backprop
        """
        # activation_error is derivative of E w.r.t Y multiplied by tanh'(XW+B)
        combined_weights = self.weights + self.m_weights[task]
        combined_bias = self.bias + self.m_bias[task]
        # y = self.activation(matmul(combined_weights.T,self.input)+combined_bias)

        # activation_error = dy/dz * dE/dy
        activation_error = hadamard(self.activation_prime(self.output),output_error)
        
        # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
        input_error = matmul(combined_weights,activation_error) #dE/dX

        weights_error = outer_product(self.input,activation_error) #dE/dW

        # update parameters with SGD
        self.weights -= shared_learning_rate * weights_error
        self.bias -= shared_learning_rate * activation_error
        self.m_weights[task] -= task_learning_rate * weights_error
        self.m_bias[task] -= task_learning_rate * activation_error
        
        return input_error


def Build_MTL_ADD_NN_Architecture(self, n_expl_vars, hidden_layer_units, n_target_units, n_tasks):
    """
    self - This should be a NeuralNetwork class object
    n_expl_vars - This should be a scalar representing the number of input neurons for the first NetworkLayer of the network
    n_target_units - This should be a scalar representing the number of output neurons for the last NetworkLayer of the network
    hidden_layer_units - This should be a list of scalars representing the number of neurons for the intermediate layers
                            between the input and output layers of the network.
    n_tasks - This should be a scalar value representing the number of tasks trained simultaneously in the network
    This function adds NetworkLayer class objects to a list called layers in the NeuralNetwork class object. If hidden_layer_units 
        is of size 2, that means there are two hidden layers. The first NetworkLayer will have n_expl_vars input neurons and 
        hidden_layer_units[0] output neurons. The seconde NetworkLayer will have hidden_layer_units[0] input neurons and 
        hidden_layer_units[1] output neurons. The third NetworkLayer will have hidden_layer_units[1] input neurons and 
        n_target_units output neurons
    """
    for layer in range(len(hidden_layer_units)):
        if layer == 0:
            self.add(MTL_ADD_Network_Layer(n_expl_vars, hidden_layer_units[layer],n_tasks))
        else:
            self.add(MTL_ADD_Network_Layer(hidden_layer_units[layer-1], hidden_layer_units[layer]),n_tasks)

    if len(hidden_layer_units) == 0:
        last_layer_units = n_expl_vars 
    else:
        last_layer_units = hidden_layer_units[-1]

    self.add(MTL_ADD_Network_Layer(last_layer_units, n_target_units, n_tasks, activation=sigmoid))


class MTL_ADD_NN:
    """
    This class defines the entire scope of a simple fully connected artificial neural network using our product mask
        ideas.
    """
    def __init__(self, n_expl_vars, hidden_layer_units, n_target_units, n_tasks):
        """
        n_expl_vars - This should be a scalar representing the number of input neurons for the first PMASKNetworkLayer of the network
        n_target_units - This should be a scalar representing the number of output neurons for the last PMASKNetworkLayer of the network
        hidden_layer_units - This should be a list of scalars representing the number of neurons for the intermediate layers
                                between the input and output layers of the network.
        n_tasks - This should be a scalar value representing the number of tasks trained simultaneously in the network
        loss - this should be a function which you measure the loss of examples trained on your network
        loss_prime - this should be a function that is the derivative of a loss function
        """
        self.layers = []
        self.loss = binary_cross_entropy
        self.loss_prime = cross_entropy_prime
        self.n_tasks = n_tasks
        Build_MTL_ADD_NN_Architecture(self, n_expl_vars, hidden_layer_units, n_target_units, self.n_tasks)

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        """
        input_data - This should be a scalar matrix of size (n_observations,n_expl_vars)
        This function performs the forward propagation pass over the network for all input_data observations
            and classifies each example using multi-label classification
        """
        results = []
        
        for x in input_data:
            task_num = x[0]
            output = x[1:]
            for layer in self.layers:
                output = layer.forward_propagation(output,task_num)
            results.append(multilabel_classification(output))
        
        return results

    def test(self,x_test,y_test):
        """
        x_test - This should be a scalar matrix of size (n_observations,n_expl_vars) (may or may not include task-bit)
        y_test - This should be a scalar matrix of size (n_observations,n_target_units)
        This function takes in explanatory dataset and response dataset and outputs the accuracy
            when the explanatory dataset is tested against the response dataset using the parameters
            of the network
        """
        y_pred = self.predict(x_test)
        return accuracy(y_test,y_pred)

    # train the network
    def fit(self, x_train, y_train, n_iterations, shared_learning_rate=0.001,task_learning_rate=0.0005,testing=250):
        """
        x_train - This should be a scalar matrix of size (n_observations,n_expl_vars) representing the explanatory
                    training dataset
        y_train - This should be a scalar matrix of size (n_observations,n_expl_vars) representing the response
                    training dataset
        n_iterations - This should be a scalar that represents how many examples from the training dataset that you train
                        your network on
        learning_rate - This should be a scalar that represents how fast you should update the parameters of your network
        testing - This should be a scalar that represents how frequently you see the efficacy of your network during training
        """
        n_observations = np.shape(x_train)[0]
        # batch_size = round(n_observations*gradient_batch_prop)
        acc_scores = []
        cross_entropy_scores = []
        
        for i in range(n_iterations):
            true_vals = []
            predicted_vals = []
#             cross_entropy = 0
            
            #batch_size = round(n_observations*(max(0.1,i/n_iterations))) dynamic batch size
            #learning_rate = 0.1*(1 - i/n_iterations) dynamic learning rate
            
            # sample_observations = random.sample(range(n_observations), batch_size)
            """
            Get a random observation to train through the network
            """
            obs = np.random.randint(0,len(x_train))
            task_num = x_train[obs,0]
            x = x_train[obs,1:]
            y = y_train[obs]
            # for x,y in zip(x_train,y_train):
            
            """
            Do the forward propagation pass
            """
            output = x
            for layer in self.layers:
                output = layer.forward_propagation(output, task_num)
            
            # true_vals.append(y)
            # predicted_vals.append(multilabel_classification(output))
#                 cross_entropy += max(0,self.loss(y_train[obs], output)) #the max is there to account for infinite loss warning
            output_error = self.loss_prime(y, output)
            # output_error = output_error.reshape(len(output_error),1)
            
            """
            Do the backward propagation pass
            """
            for layer in reversed(self.layers):
                output_error = layer.backward_propagation(output_error, shared_learning_rate, task_learning_rate, task_num)
                
            # print(x,output,predicted_vals[-1:],y)
            
#             cross_entropy_scores.append(cross_entropy/len(sample_observations))
            if i % testing == 0:
                print("Iteration: ",i)
                acc_scores.append(self.test(x_train,y_train))
        
        # fig, axs = plt.subplots(1,1,figsize=(10,10))        
        # plt.plot(np.linspace(0,n_iterations,len(acc_scores)), acc_scores)

        return np.linspace(0,n_iterations,len(acc_scores)), acc_scores