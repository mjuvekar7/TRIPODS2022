import numpy as np
import random
import matplotlib.pyplot as plt
from mtl_framework_nn import *
from functions import *

"""Markdown

"""

def main():
    random.seed(1)

    n_bits_each = 1
    hidden_layer_units = [5]
    n_iterations = 10000
    s_lr = 0.2
    t_lr = 0.1

    n_runs = 10
    test_freq = 1000
    n_accuracies = int(n_iterations/test_freq)


    X,Y = generateCircuitData(n_bits_each,["OR","AND","XOR","NOR","NAND"])
    # X = X[:,1:]
    n_tasks = 5

    n_observations, n_variables = np.shape(X)
    n_target_units = n_bits_each


    a = np.zeros((n_accuracies, n_runs))
    for n in range(n_runs):
        NeuralNet = MTL_ADD_NN(n_variables-1, hidden_layer_units, n_target_units, n_tasks)
        _, accuracies = NeuralNet.fit(X, Y, n_iterations, shared_learning_rate=s_lr,task_learning_rate=t_lr,testing=test_freq)

        for i, ai in enumerate(accuracies):
            a[i, n] = ai

    a_ste = []
    a_mean = []
    for row in range(len(a)):
        a_mean.append(np.mean(a[row,:]))
        a_ste.append(3*np.std(a[row,:])/np.sqrt(n_runs)) #68% CI is 2*std

    x_values = np.arange(0,n_iterations,test_freq)
    fig, axs = plt.subplots(1,2, figsize=(18,9))
    axs[0].plot(x_values, a_mean)
    axs[0].errorbar(x_values, a_mean, yerr = a_ste, fmt ='o',ms=3)
    axs[0].hlines(1,0,n_iterations,ls='-.')
    axs[0].set_ylim(0,1.05)
    axs[0].set_title("Mean Accuracy of Multi-task Network")
    axs[0].set_xlabel("Number of timesteps")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(("Tasks Accuracy","100% Accuracy Refline","99.7% CI Errorbars"),loc="lower right")
    for p in a.T:
        axs[1].plot(x_values, p)
    axs[1].set_ylim(0,1.05)
    axs[1].set_title("Individual Runs of the Multi-task Network")
    axs[1].set_xlabel("Number of timesteps")
    axs[1].set_ylabel("Accuracy")
    plt.savefig("test_run1.png")
    # fig.show()


if __name__ == "__main__":
    main()