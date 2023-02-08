import numpy as np
from numpy import random
from matplotlib import pyplot as plt

import HELP


class TNetwork:
    def __init__(self, layer_sizes, hidden_function, output_function, num_tasks):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.num_tasks = num_tasks
        self.layers = self.build(hidden_function, output_function)

    def build(self, hidden_function, output_function):
        input_lyr = [TLayer(self.layer_sizes[0], "Identity", self.num_tasks)]
        hidden_lyrs = [TLayer(self.layer_sizes[i], hidden_function, self.num_tasks) for i in range(1, self.num_layers - 1)]
        output_lyr = [TLayer(self.layer_sizes[self.num_layers - 1], output_function, self.num_tasks)]
        return input_lyr + hidden_lyrs + output_lyr

    def initialize(self, low, high):
        for i in range(1, self.num_layers):
            prev_lyr = self.layers[i - 1]
            lyr = self.layers[i]
            lyr.biases = random.random([lyr.size, 1]) * (high - low) + low
            lyr.weights = random.random([prev_lyr.size, lyr.size]) * (high - low) + low
            lyr.weight_diffs = [np.zeros(np.shape(lyr.weights))] + [random.random(np.shape(lyr.weights)) * (high - low) + low for _ in range(1, self.num_tasks)]
            lyr.bias_diffs = [np.zeros(np.shape(lyr.biases))] + [random.random(np.shape(lyr.biases)) * (high - low) + low for _ in range(1, self.num_tasks)]

    def forward(self, x, task_num):
        self.layers[0].inputs = x
        self.layers[0].activate()
        for i in range(1, self.num_layers):
            prev_lyr = self.layers[i - 1]
            lyr = self.layers[i]
            lyr.inputs = np.matmul((lyr.weights + lyr.weight_diffs[task_num]).T, prev_lyr.outputs) + (lyr.biases + lyr.bias_diffs[task_num])
            lyr.activate()

    def backward(self, loss_function, learn_rate, y, task_num):
        i = self.num_layers - 1
        while i > 0:
            prev_lyr = self.layers[i - 1]
            lyr = self.layers[i]
            if i == self.num_layers - 1:
                lyr.calculate_output_errors(loss_function, y, task_num)
            else:
                next_lyr = self.layers[i + 1]
                lyr.calculate_errors(next_lyr, task_num)
            lyr.gradient_descent(learn_rate, prev_lyr, task_num)
            i = i - 1

    def train(self, x_trains, y_trains, loss_function, learn_rates, num_times_list):
        for t in range(self.num_tasks):
            x_train = x_trains[t]
            y_train = y_trains[t]
            learn_rate = learn_rates[t]
            num_samples = len(x_train)
            for _ in range(num_times_list[t]):
                for j in range(num_samples):
                    self.forward(x_train[j], t)
                    self.backward(loss_function, learn_rate, y_train[j], t)

    def test(self, x_tests, y_tests, thresh):
        task_accs = []
        for t in range(self.num_tasks):
            raw_outputs = []
            classified_outputs = []
            x_test = x_tests[t]
            y_test = y_tests[t]
            num_samples = len(x_test)
            count = 0
            for j in range(num_samples):
                self.forward(x_test[j], t)
                raw_outputs.append(self.layers[self.num_layers - 1].outputs)
                classified_outputs.append(HELP.classify(self.layers[self.num_layers - 1].outputs, thresh))
                if np.array_equal(y_test[j], classified_outputs[j]):
                    count = count + 1
            task_accs.append(count / num_samples)
        return task_accs

    def ttp(self, x_trains, y_trains, loss_function, learn_rates, num_times_list, x_tests, y_tests, thresh):
        task_accs = []
        time_accs = []
        plt.figure(figsize=(10, 7.5))

        for t in range(self.num_tasks):
            one_time = [1 if m == t else 0 for m in range(self.num_tasks)]
            shift = sum(num_times_list[0:t])
            x_axis = [m + 1 + shift for m in range(num_times_list[t])]

            for i in range(num_times_list[t]):
                self.train(x_trains, y_trains, loss_function, learn_rates, one_time)
                time_accs.append(self.test(x_tests, y_tests, thresh)[t])
            task_accs.append(time_accs)

            plt.plot(x_axis, time_accs)
            if t != 0:
                plt.axvline(x=shift, c='b')
            time_accs = []

        plt.xlabel("Number of Times Model Saw Training Set")
        plt.ylabel("Network Accuracy")
        plt.xlim(-1, sum(num_times_list) + 2)
        plt.ylim(0, 1.05)
        plt.yticks([0.1 * i for i in range(11)])
        plt.show()


class TLayer:
    def __init__(self, size, function, num_tasks):
        self.size = size
        self.function = function
        self.biases = np.zeros([0, 0])
        self.weights = np.zeros([0, 0])
        self.errors = np.zeros([0, 0])
        self.inputs = np.zeros([0, 0])
        self.outputs = np.zeros([0, 0])
        self.d_outputs = np.zeros([0, 0])
        self.weight_diffs = [np.zeros([0, 0]) for _ in range(num_tasks)]
        self.bias_diffs = [np.zeros([0, 0]) for _ in range(num_tasks)]
        self.diff_errors = [np.zeros([0, 0]) for _ in range(num_tasks)]
        self.num_tasks = num_tasks

    def activate(self):
        if self.function == "Sigmoid":
            self.outputs = 1 / (1 + np.exp(-self.inputs))
            self.d_outputs = np.multiply(self.outputs, 1 - self.outputs)
        else:
            self.outputs = np.copy(self.inputs)
            self.d_outputs = np.ones([self.size, 1])

    def calculate_errors(self, next_lyr, task_num):
        if task_num == 0:
            self.errors = np.multiply(np.matmul(next_lyr.weights, next_lyr.errors), self.d_outputs)
        else:
            next_weight_diffs = next_lyr.weight_diffs[task_num]
            next_weights = next_lyr.weights
            next_diff_errors = next_lyr.diff_errors[task_num]
            self.diff_errors[task_num] = np.multiply(np.matmul(next_weights + next_weight_diffs, next_diff_errors), self.d_outputs)

    def calculate_output_errors(self, loss_function, y, task_num):
        if loss_function == "MSE":
            d_loss = self.outputs - y
        else:
            d_loss = np.zeros(np.shape(self.outputs))

        if task_num == 0:
            self.errors = np.multiply(d_loss, self.d_outputs)
        else:
            self.diff_errors[task_num] = np.multiply(d_loss, self.d_outputs)

    def gradient_descent(self, learn_rate, prev_lyr, task_num):
        if task_num == 0:
            self.weights = self.weights - learn_rate * np.matmul(prev_lyr.outputs, self.errors.T)
            self.biases = self.biases - learn_rate * self.errors
        else:
            self.weight_diffs[task_num] = self.weight_diffs[task_num] - learn_rate * np.matmul(prev_lyr.outputs, self.diff_errors[task_num].T)
            self.bias_diffs[task_num] = self.bias_diffs[task_num] - learn_rate * self.diff_errors[task_num]
