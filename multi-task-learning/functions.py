import numpy as np
import math
import random
import itertools



"""Markdown
# Circuit Data Generation Functions
    The functions below generate data representing input and output of novel circuit 
    operations. Below contains functionality for OR, XOR, NOR, AND, and NAND circuit 
    operations. We can generate data with any number of bits, not just two like in 
    classical examples.

    The common characteristic of these operators is they take two binary 
    vectors of any length and do elementwise comparisons to obtain a single vector half 
    the size of the two concatenated vectors. If B is the set [0,1] and n is the length of
    one of the binary vectors, then f: B^n * B^n -> B^n, where * is the circuit operator.
"""

def OR(X, Y):
    # when both x and y equal 0, then result is 0. Otherwise, result is 1
    return [0 if x == y == 0 else 1 for x,y in zip(X,Y)]

def XOR(X, Y):
    # when only x or y equals 1, then result is 1. Otherwise, result is 0
    return [(x + y) % 2 for x,y in zip(X,Y)]

def NOR(X, Y):
    # when both x and y equal 0, then result is 1. Otherwise, result is 0
    return [1 if x == y == 0 else 0 for x,y in zip(X,Y)]

def AND(X, Y):
    # when both x and y equal 1, then result is 1. Otherwise, result is 0
    return [1 if x == y == 1 else 0 for x,y in zip(X,Y)]

def NAND(X, Y):
    # when both x and y equal 1, then result is 0. Otherwise, result is 1
    return [0 if x == y == 1 else 1 for x,y in zip(X,Y)] 


# Task 0
def getORData(n_bits, task_num):
    bit_lists = [[0,1] for bit in range(2*n_bits)] # this prepares us to take the cartesian product to obtain the set of all 0,1 vectors of the length 2*n_bits
    X = [[task_num] + list(element) for element in itertools.product(*bit_lists)] # this obtains the entire set of 0 and 1 vectors of length 2*n_bits and appends the task number to the front of each example
    Y = [OR(x[1:n_bits+1],x[n_bits+1:]) for x in X] # this calculates the result of each example
    return np.array(X),np.array(Y)

# Task 1
def getXORData(n_bits, task_num):
    bit_lists = [[0,1] for bit in range(2*n_bits)]
    X = [[task_num] + list(element) for element in itertools.product(*bit_lists)]
    Y = [XOR(x[1:n_bits+1],x[n_bits+1:]) for x in X]
    return np.array(X),np.array(Y)

# Task 0
def getNORData(n_bits, task_num):
    bit_lists = [[0,1] for bit in range(2*n_bits)]
    X = [[task_num] + list(element) for element in itertools.product(*bit_lists)]
    Y = [NOR(x[1:n_bits+1],x[n_bits+1:]) for x in X]
    return np.array(X),np.array(Y)

# Task 1
def getANDData(n_bits, task_num):
    bit_lists = [[0,1] for bit in range(2*n_bits)]
    X = [[task_num] + list(element) for element in itertools.product(*bit_lists)]
    Y = [AND(x[1:n_bits+1],x[n_bits+1:]) for x in X]
    return np.array(X),np.array(Y)

# Task 4
def getNANDData(n_bits, task_num):
    bit_lists = [[0,1] for bit in range(2*n_bits)]
    X = [[task_num] + list(element) for element in itertools.product(*bit_lists)]
    Y = [NAND(x[1:n_bits+1],x[n_bits+1:]) for x in X]
    return np.array(X),np.array(Y)

def generateCircuitData(num_bits, operator_list):
    """
    num_bits - the length of each input bit vector
    operator_list - this is the list of operators we want to obtain data for
    This function generates circuit operator data for the given operators, and returns a numpy input dataset, X, of size (n,p)
        where n is the number of observations (n_tasks*2^(2*num_bits)) and p is 2*num_bits + 1, and numpy output dataset, Y, of size
        (n,) where n is the same as above
    """
    operators = ["OR","XOR","NOR","AND","NAND"]
    functions = [getORData, getXORData, getNORData, getANDData, getNANDData]
    X = []
    Y = []
    task_num = 0
    for operator in operator_list:
        if operator in operators:
            func = functions[operators.index(operator)]
            op_train = func(num_bits, task_num)
            X = list(X) + list(op_train[0])
            Y = list(Y) + list(op_train[1])
            task_num+=1
    return np.array(X),np.array(Y)

"""Markdown
# Matrix Product Functions
    The functions below are various matrix products. Using x*y is ambiguous so it was 
    better for us to define each case.
"""

def outer_product(x,y):
    # if you have an (n,) and (m,) vector, this function multiplies them such that the output is an (n,m) matrix
    return np.reshape([xi*yj for xi in x for yj in y],(len(x),len(y)))

def matmul(x,y):
    # if you have an (n,p) and (p,k) matrix, this function multiplies them such that the output is an (n,k) matrix
    return np.matmul(x,y)

def hadamard(x,y):
    # if you have two (n,k) matrices, this function performs the element-wise product of the two matrices
    return x*y

"""Markdown
# Activation Functions
"""

def sigmoid(z):
    # this function takes a linear combination sum as input (z = w.T*x + b)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(y):
    # this function takes a sigmoid output as input (y = f(z) where f is sigmoid)
    return y * (1 - y)

"""Markdown
# Classification Functions
    TODO:
    We could also build a function that considers partial example accuracy instead of 
    all or nothing accuracy
"""

def multilabel_classification(PROBS, output_dim = 1, thresh = 0.5):
    return np.reshape([1 if prob > thresh else 0 for prob in PROBS],(len(PROBS),))

def classify(PROBS, thresh = 0.5):
    return np.array([1 if prob > thresh else 0 for prob in PROBS])

def accuracy(y_test,y_pred):
    #.all means elements at corresponding indices of pred and test must equal for all element pairs
    return np.mean([1 if (pred == test).all() else 0 for pred,test in zip(y_pred,y_test)])

"""Markdown
# Loss Functions
"""

def binary_cross_entropy(true_val, y_hat, e=0.0000001):
    return -true_val*np.log(y_hat+e) - (1-true_val)*np.log(1-y_hat+e)

def multilabel_cross_entropy(true_val,y_hat, e=0.00000001):
    return np.sum([binary_cross_entropy(t,y) for t,y in zip(true_val,y_hat)])

def cross_entropy_prime(true_val,y_hat, e=0.0):
    return -true_val/(y_hat+e) + (1-true_val)/(1-y_hat+e)

def mse(true_val,y_hat):
    return (1/2)*(y_hat - true_val)**2

def mse_prime(true_val,y_hat):
    return y_hat - true_val

"""Markdown
# Miscellaneous Functions
"""