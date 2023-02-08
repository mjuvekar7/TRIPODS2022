import numpy as np
from numpy import random
import math
from PIL import Image, ImageDraw


class Network:
    # Network constructor.
    def __init__(self, layer_sizes, loss_function, hidden_function, output_function, learn_rate, num_tasks):
        # Initialize fields.
        self.layer_sizes = layer_sizes
        self.loss_function = loss_function
        self.learn_rate = learn_rate
        self.L = len(layer_sizes)

        if num_tasks is None:
            self.num_tasks = 1
        else:
            self.num_tasks = num_tasks

        self.Layers = []

        # Builds the Network.
        self.Build(hidden_function, output_function)

    # Builds the Layers of the Network.
    def Build(self, hidden_function, output_function):
        for i in range(self.L):
            if i == 0:
                function = "Identity"
            elif i < self.L - 1:
                function = hidden_function
            else:
                function = output_function
            self.addLayer(Layer(self.layer_sizes[i], function, self.num_tasks))

    # Tests the Network.
    def Test(self, xTest):
        yTest = []
        for i in range(len(xTest)):
            self.Forward(xTest[i])
            yTest.append(self.Layer(self.L - 1).getActivations())
        return yTest

    # Trains the Network.
    def Train(self, xTrain, yTrain):
        for i in range(len(xTrain)):
            self.Forward(xTrain[i])
            self.Backward(yTrain[i])

    # Back-propagation.
    def Backward(self, Y):
        i = self.L - 1
        while i > 0:
            Lyr = self.Layer(i)
            for j in range(Lyr.size):
                Nrn = Lyr.Neuron(j)
                if i == self.L - 1:
                    Nrn.calculateOutputError(Y[j, 0], self.loss_function)
                else:
                    Nrn.calculateError(j)
                Nrn.descent(self.learn_rate)
            i = i - 1

    # Feed-forward.
    def Forward(self, X):
        self.Layer(0).setInputs(X)
        self.Layer(0).activate()

        for i in range(1, self.L):
            Lyr = self.Layer(i)
            for j in range(Lyr.size):
                Nrn = Lyr.Neuron(j)
                W = Nrn.Weights.T
                A = Nrn.getPreviousActivations()
                B = np.full([1, 1], Nrn.bias)
                Z = np.matmul(W, A) + B
                Nrn.setInput(Z[0, 0])
                Nrn.activate()

    # Methods for Network construction and initialization.
    def attach(self, fl, fn, tl, tn):
        fromNrn = self.Layer(fl).Neuron(fn)
        toNrn = self.Layer(tl).Neuron(tn)
        fromNrn.addOut(toNrn)
        toNrn.addIn(fromNrn)

    def attachLayer(self, fl, tl):
        for i in range(self.Layer(fl).size):
            for j in range(self.Layer(tl).size):
                self.attach(fl, i, tl, j)

    def attachSubsets(self, fl, fSet, tl, tSet):
        for i in range(len(fSet)):
            for j in range(len(tSet)):
                self.attach(fl, fSet[i], tl, tSet[j])

    def initialize(self):
        for i in range(1, self.L):
            Lyr = self.Layer(i)
            for j in range(Lyr.size):
                Nrn = Lyr.Neuron(j)
                Nrn.setWeights(random.random([len(Nrn.In), 1]))
                Nrn.setBias(random.rand())

    # Getters.
    def Layer(self, i):
        return self.Layers[i]

    # Adders.
    def addLayer(self, Lyr):
        self.Layers.append(Lyr)

    # Draws Network.
    def draw(self, width, height):
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)
        max_size = max(self.layer_sizes)

        # Calculates scaling values.
        w = width / self.L
        h = height / max_size
        x = w / 2
        if h / 4 > 30:
            delta = 30
        else:
            delta = h / 4

        # Draws Neurons.
        for Lyr in self.Layers:
            y = (h / 2)
            for Nrn in Lyr.Neurons:
                draw.ellipse((x - delta, y - delta, x + delta, y + delta), fill="white")
                Nrn.setLocation(x, y)
                y = y + h
            x = x + w

        # Draws edges.
        for Lyr in self.Layers:
            for Nrn in Lyr.Neurons:
                for outNrn in Nrn.Out:
                    draw.line((Nrn.location, outNrn.location), fill="white")

        image.show()

    # Prints Network.
    def print(self):
        for i in range(self.L):
            self.Layer(i).print(i)


class MTNetwork(Network):
    # Multi-Task Network Constructor.
    def __init__(self, layer_sizes, loss_functions, loss_weights, hidden_function, output_function, learn_rate,
                 num_tasks):
        # Initialize fields.
        super().__init__(layer_sizes, "", hidden_function, output_function, learn_rate, num_tasks)
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights

    # Tests the Multi-Task Network.
    def MTest(self, xTaskTest, yTaskTest, num_times):
        task_accuracies = []
        raw_task_outputs = []
        classified_task_outputs = []

        for k in range(num_times):
            counts = [0 for i in range(self.num_tasks)]
            for i in range(self.num_tasks):
                xTest = xTaskTest[i]
                yTest = yTaskTest[i]
                raw_outputs = []
                classified_outputs = []

                for j in range(len(xTest)):
                    self.Forward(xTest[j])
                    raw_output = self.Layer(self.L - 1).getActivations()
                    classified_output = classify(raw_output, 0.5)
                    raw_outputs.append(raw_output)
                    classified_outputs.append(classified_output)
                raw_task_outputs.append(raw_outputs)
                classified_task_outputs.append(classified_outputs)

                for j in range(len(xTest)):
                    if np.array_equal(classified_outputs[j], yTest[j]):
                        counts[i] = counts[i] + 1
                task_accuracies.append(counts[i] / len(xTest))

        print("Task Accuracies:")
        for i in range(self.num_tasks):
            print("Task " + str(i) + ": " + str(task_accuracies[i]*100) + "% with " + str(num_times * len(xTaskTest[i])) + " test samples.")
        print("")

        return [raw_task_outputs, classified_task_outputs, task_accuracies]

    # Trains the Multi-Task Network with independent losses.
    # xTaskTrain is a list of xTrain training sets. Each xTrain corresponds to one of the tasks.
    # yTaskTrain is a list of yTrain training labels. Each yTrain corresponds to one of the tasks.
    def MTrain(self, xTaskTrain, yTaskTrain, num_to_train, getAccuracy = 500):
        training_task_accuracies = []
        for i in range(num_to_train):
            
            #task_num = random.randint(0, self.num_tasks-1, [1, 1])[0, 0]
            # sample_num = random.randint(0, len(xTaskTrain[task_num])-1, [1, 1])[0, 0]
            for task_num in range(self.num_tasks):
                sample_num = random.randint(0, len(xTaskTrain[task_num])-1, [1, 1])[0, 0]
                x_sample = xTaskTrain[task_num][sample_num]
                y_sample = yTaskTrain[task_num][sample_num]
                self.Forward(x_sample)
                self.MBackward(y_sample, task_num)
            if i%getAccuracy == 0:
                _,_,task_accuracies = self.MTest(xTaskTrain, yTaskTrain, 1)
                training_task_accuracies.append(task_accuracies)
        return training_task_accuracies

    # Trains the Multi-Task Network with a joint loss function.
    def MJointTrain(self, xTaskTrain, yTaskTrain, num_to_train, getAccuracy = 500):
        training_task_accuracies = []
        for i in range(num_to_train):
            task_num = random.randint(0, self.num_tasks-1, [1, 1])[0, 0]
            sample_num = random.randint(0, len(xTaskTrain[task_num])-1, [1, 1])[0, 0]
            x_sample = xTaskTrain[task_num][sample_num]
            y_sample = yTaskTrain[task_num][sample_num]
            self.Forward(x_sample)
            self.MJointBackward(y_sample)
            if i%getAccuracy == 0:
                _,_,task_accuracies = self.MTest(xTaskTrain, yTaskTrain, 1)
                training_task_accuracies.append(task_accuracies)
        return training_task_accuracies

    # Back-propagation with individual loss functions for each task.
    def MBackward(self, Y, task_num):
        i = self.L - 1
        while i > 0:
            Lyr = self.Layer(i)
            for j in range(Lyr.size):
                Nrn = Lyr.Neuron(j)
                if i == self.L - 1:
                    Nrn.calculateOutputError(Y[j, 0], self.loss_functions[task_num])
                else:
                    Nrn.calculateError(j)
                Nrn.descent(self.learn_rate)
            i = i - 1

    # Back-propagation with a joint loss function, combining losses for each task.
    def MJointBackward(self, Y):
        i = self.L - 1
        while i > 0:
            Lyr = self.Layer(i)
            for j in range(Lyr.size):
                Nrn = Lyr.Neuron(j)
                if i == self.L - 1:
                    Nrn.calculateJointOutputError(Y[j, 0], self.loss_functions, self.loss_weights)
                else:
                    Nrn.calculateError(j)
                Nrn.descent(self.learn_rate)
            i = i - 1


class Layer:
    # Layer constructor.
    def __init__(self, size, function, num_tasks):
        self.size = size
        self.Neurons = [Neuron(function, num_tasks) for _ in range(size)]

    # Calculates the activations for every Neuron in the Layer.
    def activate(self):
        for Nrn in self.Neurons:
            Nrn.activate()

    # Getters.
    def Neuron(self, i):
        return self.Neurons[i]

    def getActivations(self):
        activations = [self.Neuron(i).activation for i in range(self.size)]
        return np.reshape(activations, [len(activations), 1])

    # Setters.
    def setInputs(self, X):
        for i in range(np.shape(X)[0]):
            self.Neuron(i).setInput(X[i, 0])

    # Prints Layer.
    def print(self, i):
        print("Layer " + str(i) + ":")
        for j in range(self.size):
            self.Neuron(j).print(j)
        print("")


class Neuron:
    # Neuron constructor.
    def __init__(self, function, num_tasks):
        self.function = function
        self.bias = 0
        self.Weights = np.zeros([0, 0])
        self.input = 0
        self.activation = 0
        self.error = 0
        self.dcost = np.zeros([num_tasks, 1])
        self.dactivation = 0
        self.In = []
        self.Out = []
        self.location = (0, 0)

    # Updates weights and biases with gradient descent.
    def descent(self, learn_rate):
        new_bias = self.bias - learn_rate * self.error
        self.setBias(new_bias)
        new_Weights = self.Weights - self.getPreviousActivations() * self.error * learn_rate
        self.setWeights(new_Weights)

    # Calculates the error of a hidden Neuron.
    def calculateError(self, j):
        nextWeights = self.getNextWeights(j).T
        nextErrors = self.getNextErrors()
        error = np.matmul(nextWeights, nextErrors) * self.dactivation
        self.setError(error[0, 0])

    # Calculates the error of an output Neuron with a joint loss-function set-up.
    def calculateJointOutputError(self, y, loss_functions, loss_weights):
        # Calculates joint dcost.
        loss_weights = normalize(loss_weights)
        self.calculateDCosts(y, loss_functions)
        joint_dcost = np.matmul(self.dcost.T, loss_weights)[0, 0]
        error = joint_dcost * self.dactivate()
        self.setError(error)

    # Calculates the error of an output Neuron with 1 Task.
    def calculateOutputError(self, y, loss_function):
        error = self.calculateDCost(y, 0, loss_function) * self.dactivate()
        self.setError(error)

    # Calculates dC/dA.
    def calculateDCost(self, y, task_num, loss_function):
        dcost = 0
        a = self.activation
        if loss_function == "MSE":
            dcost = a - y
        self.setDCost(task_num, dcost)
        return dcost

    def calculateDCosts(self, y, loss_functions):
        for i in range(len(loss_functions)):
            self.calculateDCost(y, i, loss_functions[i])

    # Calculates dA/dZ.
    def dactivate(self):
        a = self.activation
        dactivation = 0
        if self.function == "Identity":
            dactivation = 1
        elif self.function == "Sigmoid":
            dactivation = a * (1 - a)
        self.setDActivation(dactivation)
        return dactivation

    # Calculates the activation.
    def activate(self):
        z = self.input
        if self.function == "Identity":
            self.setActivation(z)
        elif self.function == "Sigmoid":
            self.setActivation(1 / (1 + math.exp(-z)))

    # Adders.
    def addIn(self, Nrn):
        self.In.append(Nrn)

    def addOut(self, Nrn):
        self.Out.append(Nrn)

    # Setters.
    def setLocation(self, x, y):
        self.location = (x, y)

    def setBias(self, bias):
        self.bias = bias

    def setError(self, error):
        self.error = error

    def setInput(self, x):
        self.input = x

    def setActivation(self, activation):
        self.activation = activation

    def setDCost(self, i, dcost):
        self.dcost[i, 0] = dcost

    def setDActivation(self, dactivation):
        self.dactivation = dactivation

    def setFunction(self, function):
        self.function = function

    def setWeights(self, Weights):
        self.Weights = Weights

    # Getters.
    def getPreviousActivations(self):
        previousActivations = [Nrn.activation for Nrn in self.In]
        return np.reshape(previousActivations, [len(previousActivations), 1])

    def getNextWeights(self, j):
        nextWeights = [Nrn.Weights[j, 0] for Nrn in self.Out]
        return np.reshape(nextWeights, [len(nextWeights), 1])

    def getNextErrors(self):
        nextErrors = [Nrn.error for Nrn in self.Out]
        return np.reshape(nextErrors, [len(nextErrors), 1])

    # Prints Neuron.
    def print(self, j):
        print("Neuron " + str(j) + ": (Bias = " + str("{:.2f}".format(self.bias)) + ", Error = " + str(
            "{:.2f}".format(self.error)) + ", Input = " + str(
            "{:.2f}".format(self.input)) + ", Activation = " + str("{:.2f}".format(self.activation)) + ")")


# ----------------------------------------------------------------------------------------------------------------------
# Converts a base 10 number n into a base b vector.
def convert(n, b, size):
    vector = []
    i = 0
    while n > 0:
        r = n % b
        vector = [r] + vector
        n = math.floor(n / b)
        i = i + 1
    while i < size:
        vector = [0] + vector
        i = i + 1
    return np.reshape(vector, [len(vector), 1])


# Normalizes a list of numbers.
def normalize(vector):
    add = 0
    for x in vector:
        add = add + x * x
    length = math.sqrt(add)
    normed_vector = vector / length
    return normed_vector


# OR's 2 different 2-bit vectors together.
def OR(X, Y):
    Z = []
    for i in range(np.shape(X)[0]):
        if X[i, 0] == 0 and Y[i, 0] == 0:
            Z.append(0)
        else:
            Z.append(1)
    return np.reshape(Z, [len(Z), 1])


# XOR's 2 different 2-bit vectors together.
def XOR(X, Y):
    Z = []
    for i in range(np.shape(X)[0]):
        Z.append((X[i, 0] + Y[i, 0]) % 2)
    return np.reshape(Z, [len(Z), 1])


# Task 0
def getORTrain(num_bits):
    xTrain = []
    yTrain = []
    for i in range(2**num_bits):
        X = np.concatenate((np.zeros([1, 1]), convert(i, 2, 2**num_bits)))
        A = X[1:2**(num_bits-1)+1, :]
        B = X[2**(num_bits-1)+1:2**num_bits+1, :]
        Y = OR(A, B)
        xTrain.append(X)
        yTrain.append(Y)
    return [xTrain, yTrain]


# Task 1
def getXORTrain(num_bits):
    xTrain = []
    yTrain = []
    for i in range(2**num_bits):
        X = np.concatenate((np.ones([1, 1]), convert(i, 2, 2**num_bits)))
        A = X[1:2**(num_bits-1)+1, :]
        B = X[2**(num_bits-1)+1:2**num_bits+1, :]
        Y = XOR(A, B)
        xTrain.append(X)
        yTrain.append(Y)
    return [xTrain, yTrain]


# Collects OR/XOR training data.
def getORXORTrain(num_bits):
    ORTrain = getORTrain(num_bits)
    XORTrain = getXORTrain(num_bits)
    xTaskTrain = [ORTrain[0], XORTrain[0]]
    yTaskTrain = [ORTrain[1], XORTrain[1]]
    return [xTaskTrain, yTaskTrain]


# Classifies outputs.
def classify(vector, thresh):
    classified_vector = []
    for i in range(np.shape(vector)[0]):
        if vector[i, 0] >= thresh:
            classified_vector.append(1)
        else:
            classified_vector.append(0)
    return np.reshape(classified_vector, [len(classified_vector), 1])
# ----------------------------------------------------------------------------------------------------------------------
