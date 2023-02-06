import numpy as np
from functions import *

"""Markdown

"""
class NetworkLayer:
    """
    This class defines an individual fully connected layer's connection to another layer. In this implementation,
        NetworkLayer defines the connection between two layers (an input layer and an output layer). We also utilize
        matrix calculations to perform forward and backward propagations. You'll end up keeping a list of network
        layers which represents the full network. The output layer in one NetworkLayer becomes the input layer in the
        next NetworkLayer.
    """
    def __init__(self, input_size, output_size, activation=sigmoid, activation_prime=sigmoid_prime):
        """
        input_size - This is a scalar value representing the number of input neurons to include
        output_size - This is a scalar value representing the number of output neurons to include
        activation - This is a function representing what function to use to activate the linear combination of the 
                        layers connections and the input layers input
        activation_prime - This is a function representing what function derivative to use in backpropagation

        This function initializes a NetworkLayer by providing the number of input neurons and number of output neurons. 
            You can also provide the activation function and its derivative, but the default is sigmoid
        """
        self.weights = np.random.normal(0, 1, size=(input_size, output_size)) # randomize weights over a normal distribution
        self.bias = np.random.normal(0, 1, size=(output_size,)) # randomize bias connections over a normal distribution

        self.activation = activation
        self.activation_prime = activation_prime
        
        self.input = None
        self.output = None
        
    def forward_propagation(self, input_data):
        """
        input_data - This should be a scalar vector of size (input_size,)
        This function performs the forward propagation pass for the network layer
        """
        self.input = input_data
        self.output = self.activation(matmul(self.weights.T,self.input)+self.bias) # self.output = f(z) , z = w.T*x + b
        return self.output
        
    def backward_propagation(self, output_error, learning_rate):
        """
        output_error - This should be a scalar vector of size (output_size,) which equals dE/dy of a given NetworkLayer
        learning_rate - This should be a scalar value representing how much you want to update the weights and biases
                            on each example
        This function performs the backward propagation pass for the network layer. It takes it's NetworkLayer 
            output_error (dE/dy) as input, and calculates dE/dW, dE/dB to update the NetworkLayer's parameters, and
            it calculates dE/dx which becomes dE/dy for the next NetworkLayer visited in backprop
        """
        #dE/dy = output_error
        #dy/dz = f'(w.T*x + b)
        #dz/dB = 1
        #dz/dX = w
        #dz/dW = x
        #dE/dW = dE/dy * dy/dz * dz/dW = output_error * f'(w.T*x + b) * x
        #dE/dB = dE/dy * dy/dz * dz/dB = output_error * f'(w.T*x + b) * 1
        #dE/dX = dE/dy * dy/dz * dz/dX = output_error * f'(w.T*x + b) * w
        #x = self.input
        #w = self.weights
        #b = self.bias

        # activation_error is output_error multiplied by f'(w.T*x + b)
        y = self.activation(matmul(self.weights.T,self.input) + self.bias) #f(w.T*x + b)
        activation_error = hadamard(self.activation_prime(y),output_error) #dE/dy * dy/dz

        input_error = matmul(self.weights,activation_error) #dE/dX
        weights_error = outer_product(self.input,activation_error) #dE/dW

        # update parameters with SGD
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * activation_error
        
        #Returns input_error=dE/dX.
        return input_error



def BuildNNArchitecture(self, n_expl_vars, hidden_layer_units, n_target_units):
    """
    self - This should be a NeuralNetwork class object
    n_expl_vars - This should be a scalar representing the number of input neurons for the first NetworkLayer of the network
    n_target_units - This should be a scalar representing the number of output neurons for the last NetworkLayer of the network
    hidden_layer_units - This should be a list of scalars representing the number of neurons for the intermediate layers
                            between the input and output layers of the network.
    This function adds NetworkLayer class objects to a list called layers in the NeuralNetwork class object. If hidden_layer_units 
        is of size 2, that means there are two hidden layers. The first NetworkLayer will have n_expl_vars input neurons and 
        hidden_layer_units[0] output neurons. The seconde NetworkLayer will have hidden_layer_units[0] input neurons and 
        hidden_layer_units[1] output neurons. The third NetworkLayer will have hidden_layer_units[1] input neurons and 
        n_target_units output neurons
    """
    for layer in range(len(hidden_layer_units)):
        if layer == 0:
            self.add(NetworkLayer(n_expl_vars, hidden_layer_units[layer]))
        else:
            self.add(NetworkLayer(hidden_layer_units[layer-1], hidden_layer_units[layer]))

    # if there are no hidden layers, then n_expl_vars is the number of input neurons for the last NetworkLayer of the network
    if len(hidden_layer_units) == 0:
        last_layer_units = n_expl_vars 
    else:
        last_layer_units = hidden_layer_units[-1]

    self.add(NetworkLayer(last_layer_units, n_target_units, activation=sigmoid))


class NeuralNetwork:
    """
    This class defines the entire scope of a simple fully connected artificial neural network. If your training
        data includes the task-bit variable, then this is a task-bit network. If you remove the task-bit from the
        training data, then it's just a simple FC ANN.
    """
    def __init__(self, n_expl_vars, hidden_layer_units, n_target_units, loss=binary_cross_entropy, loss_prime=cross_entropy_prime):
        """
        n_expl_vars - This should be a scalar representing the number of input neurons for the first NetworkLayer of the network
        n_target_units - This should be a scalar representing the number of output neurons for the last NetworkLayer of the network
        hidden_layer_units - This should be a list of scalars representing the number of neurons for the intermediate layers
                                between the input and output layers of the network.
        loss - this should be a function which you measure the loss of examples trained on your network
        loss_prime - this should be a function that is the derivative of a loss function
        """
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        BuildNNArchitecture(self, n_expl_vars, hidden_layer_units, n_target_units)

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
            output = x
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(multilabel_classification(output))
        
        return results

    def test(self,x_test,y_test):
        """
        x_test - This should be a scalar matrix of size (n_observations,n_expl_vars)
        y_test - This should be a scalar matrix of size (n_observations,n_target_units)
        This function takes in explanatory dataset and response dataset and outputs the accuracy
            when the explanatory dataset is tested against the response dataset using the parameters
            of the network
        """
        y_pred = self.predict(x_test)
        return accuracy(y_test,y_pred)

    # train the network
    def fit(self, x_train, y_train, n_iterations, learning_rate=0.001,testing=250):
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
            x = x_train[obs]
            y = y_train[obs]
            # for x,y in zip(x_train,y_train):
            
            """
            Do the forward propagation pass
            """
            output = x
            for layer in self.layers:
                output = layer.forward_propagation(output)
            
            # true_vals.append(y)
            # predicted_vals.append(multilabel_classification(output))
#                 cross_entropy += max(0,self.loss(y_train[obs], output)) #the max is there to account for infinite loss warning
            output_error = self.loss_prime(y, output)
            # output_error = output_error.reshape(len(output_error),1)

            """
            Do the backward propagation pass
            """
            for layer in reversed(self.layers):
                output_error = layer.backward_propagation(output_error, learning_rate)
                
            # print(x,output,predicted_vals[-1:],y)
            
#             cross_entropy_scores.append(cross_entropy/len(sample_observations))
            if i % testing == 0:
                acc_scores.append(self.test(x_train,y_train))
        
        # fig, axs = plt.subplots(1,1,figsize=(10,10))        
        # plt.plot(np.linspace(0,n_iterations,len(acc_scores)), acc_scores)

        return np.linspace(0,n_iterations,len(acc_scores)), acc_scores