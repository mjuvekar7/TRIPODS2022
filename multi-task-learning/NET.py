import numpy as np
import HELP
from numpy import random
from matplotlib import pyplot as plt


class Network:
    def __init__(self, layer_sizes, hidden_function, output_function):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.layers = self.build(hidden_function, output_function)

    def build(self, hidden_function, output_function):
        input_lyr = [Layer(self.layer_sizes[0], "Identity")]
        hidden_lyrs = [Layer(self.layer_sizes[i], hidden_function) for i in range(1, self.num_layers - 1)]
        output_lyr = [Layer(self.layer_sizes[self.num_layers - 1], output_function)]
        return input_lyr + hidden_lyrs + output_lyr

    def initialize(self, low, high):
        for i in range(1, self.num_layers):
            prev_lyr = self.layers[i - 1]
            lyr = self.layers[i]
            lyr.biases = random.random([lyr.size, 1]) * (high - low) + low
            lyr.weights = random.random([prev_lyr.size, lyr.size]) * (high - low) + low

    def forward(self, x):
        self.layers[0].inputs = x
        self.layers[0].activate()
        for i in range(1, self.num_layers):
            prev_lyr = self.layers[i-1]
            lyr = self.layers[i]
            lyr.inputs = np.matmul(lyr.weights.T, prev_lyr.outputs) + lyr.biases
            lyr.activate()

    def backward(self, loss_function, learn_rate, y):
        i = self.num_layers - 1
        while i > 0:
            prev_lyr = self.layers[i - 1]
            lyr = self.layers[i]
            if i == self.num_layers - 1:
                lyr.calculate_output_errors(loss_function, y)
            else:
                next_lyr = self.layers[i + 1]
                lyr.calculate_errors(next_lyr)
            lyr.gradient_descent(learn_rate, prev_lyr)
            i = i - 1

    def train(self, x_train, y_train, loss_function, learn_rate, num_times):
        num_samples = len(x_train)
        for i in range(num_times):
            for j in range(num_samples):
                self.forward(x_train[j])
                self.backward(loss_function, learn_rate, y_train[j])

    def test(self, x_test, y_test, thresh):
        num_samples = len(x_test)
        raw_outputs = []
        classified_outputs = []
        count = 0
        for i in range(num_samples):
            self.forward(x_test[i])
            raw_outputs.append(self.layers[self.num_layers - 1].outputs)
            classified_outputs.append(HELP.classify(self.layers[self.num_layers - 1].outputs, thresh))
            if np.array_equal(classified_outputs[i], y_test[i]):
                count = count + 1
        return count / num_samples

    def ttp(self, x_train, y_train, loss_function, learn_rate, num_times, x_test, y_test, thresh):
        accs = []
        for i in range(num_times):
            self.train(x_train, y_train, loss_function, learn_rate, 1)
            accs.append(self.test(x_test, y_test, thresh))

        plt.figure(figsize=(10, 7.5))
        plt.plot([i + 1 for i in range(num_times)], accs)
        plt.xlabel("Number of Times Model Saw Training Set")
        plt.ylabel("Network Accuracy")
        plt.xlim(-1, num_times + 2)
        plt.ylim(0, 1.05)
        plt.yticks([0.1 * i for i in range(11)])
        plt.show()


class Layer:
    def __init__(self, size, function):
        self.size = size
        self.function = function
        self.biases = np.zeros([0, 0])
        self.weights = np.zeros([0, 0])
        self.errors = np.zeros([0, 0])
        self.inputs = np.zeros([0, 0])
        self.outputs = np.zeros([0, 0])
        self.d_outputs = np.zeros([0, 0])

    def activate(self):
        if self.function == "Sigmoid":
            self.outputs = 1 / (1 + np.exp(-self.inputs))
            self.d_outputs = np.multiply(self.outputs, 1 - self.outputs)
        else:
            self.outputs = np.copy(self.inputs)
            self.d_outputs = np.ones([self.size, 1])

    def calculate_errors(self, next_lyr):
        self.errors = np.multiply(np.matmul(next_lyr.weights, next_lyr.errors), self.d_outputs)

    def calculate_output_errors(self, loss_function, y):
        if loss_function == "MSE":
            d_loss = self.outputs - y
            self.errors = np.multiply(d_loss, self.d_outputs)

    def gradient_descent(self, learn_rate, prev_lyr):
        self.weights = self.weights - learn_rate * np.matmul(prev_lyr.outputs, self.errors.T)
        self.biases = self.biases - learn_rate * self.errors


