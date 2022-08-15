# %%
import MTNet
import numpy as np
import matplotlib.pyplot as plt
import math
import IPython.display as display

# %%
x_values = np.arange(0,8000,500)
a_mean = np.arange(0,4000,250)
b_mean = np.arange(0,2000,125)
plt.plot(x_values, a_mean)
plt.plot(x_values, b_mean)
plt.title("Multi-task (One Joint Backpropagation) Accuracy Rate")
plt.xlabel("Number of timesteps")
plt.ylabel("Accuracy")
plt.legend(("OR","XOR"))
plt.show()

# %%
ORXORTrain = MTNet.getORXORTrain(5)
xTaskTrain = ORXORTrain[0]
yTaskTrain = ORXORTrain[1]

n_runs = 25
n_train_timesteps = 6000
n_accuracies = int(n_train_timesteps/500)

# %%
a = np.zeros((n_accuracies, n_runs))
b = np.zeros((n_accuracies, n_runs))
for n in range(n_runs):
    net = MTNet.MTNetwork([33, 16], ["MSE", "MSE"], np.reshape([1, 1], [2, 1]), "Sigmoid", "Sigmoid", 0.01, 2)
    net.attachLayer(0, 1)
    net.initialize()
    task_accuracies = net.MJointTrain(xTaskTrain, yTaskTrain, n_train_timesteps) #Number of forward and backpropogations taking random samples
    for i, (ai, bi) in enumerate(task_accuracies):
        a[i, n] = ai
        b[i, n] = bi

a_std = []
a_mean = []
for row in range(len(a)):
    a_mean.append(np.mean(a[row,:]))
    a_std.append(np.std(a[row,:]))

b_std = []
b_mean = []
for row in range(len(b)):
    b_mean.append(np.mean(b[row,:]))
    b_std.append(np.std(b[row,:]))

print(a_mean, "mean of task A")
print(a_std, "STD of task A")
print(b_mean, "mean of task B")
print(b_std, "STD of task B")

x_values = np.arange(0,n_train_timesteps,500)
plt.plot(x_values, a_mean)
plt.errorbar(x_values, a_mean, yerr = a_std, fmt ='o')
plt.plot(x_values, b_mean)
plt.errorbar(x_values, b_mean, yerr = b_std, fmt = 'o')
plt.title("Multi-task (One Joint Backpropagation) Accuracy Rate")
plt.xlabel("Number of timesteps")
plt.ylabel("Accuracy")
plt.legend(("OR","XOR"))
plt.show()

# %%
a = np.zeros((n_accuracies, n_runs))
b = np.zeros((n_accuracies, n_runs))
for n in range(n_runs):
    net = MTNet.MTNetwork([33, 16], ["MSE", "MSE"], np.reshape([1, 1], [2, 1]), "Sigmoid", "Sigmoid", 0.01, 2)
    net.attachLayer(0, 1)
    net.initialize()
    task_accuracies = net.MTrain(xTaskTrain, yTaskTrain, n_train_timesteps) #Number of forward and backpropogations taking random samples
    for i, (ai, bi) in enumerate(task_accuracies):
        a[i, n] = ai
        b[i, n] = bi

a_std = []
a_mean = []
for row in range(len(a)):
    a_mean.append(np.mean(a[row,:]))
    a_std.append(np.std(a[row,:]))

b_std = []
b_mean = []
for row in range(len(b)):
    b_mean.append(np.mean(b[row,:]))
    b_std.append(np.std(b[row,:]))

print(a_mean, "mean of task A")
print(a_std, "STD of task A")
print(b_mean, "mean of task B")
print(b_std, "STD of task B")

plt.plot(x_values, a_mean)
plt.errorbar(x_values, a_mean, yerr = a_std, fmt ='o')
plt.plot(x_values, b_mean)
plt.errorbar(x_values, b_mean, yerr = b_std, fmt = 'o')
plt.title("Multi-task (Individual Task Backpropagations) Accuracy Rate")
plt.xlabel("Number of timesteps")
plt.ylabel("Accuracy")
plt.legend(("OR","XOR"))
plt.show()
