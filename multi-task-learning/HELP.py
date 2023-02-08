import numpy as np
import math


def classify(y, thresh):
    classified = []
    for num in y:
        if num >= thresh:
            classified.append(1)
        else:
            classified.append(0)
    return np.reshape(classified, np.shape(y))


def convert(n, b, size):
    z = []
    while n > 0:
        r = n % b
        z = [r] + z
        n = math.floor(n / b)
    z = [0 for _ in range(size - len(z))] + z
    return np.reshape(z, [len(z), 1])


def XOR(a, b):
    n = np.shape(a)[0]
    z = [(a[i, 0] + b[i, 0]) % 2 for i in range(n)]
    return np.reshape(z, np.shape(a))


def OR(a, b):
    n = np.shape(a)[0]
    z = [0 if a[i, 0] == 0 and b[i, 0] == 0 else 1 for i in range(n)]
    return np.reshape(z, np.shape(a))


def AND(a, b):
    n = np.shape(a)[0]
    z = [0 if a[i, 0] == 0 or b[i, 0] == 0 else 1 for i in range(n)]
    return np.reshape(z, np.shape(a))


def NAND(a, b):
    n = np.shape(a)[0]
    z = [1 if a[i, 0] == 0 or b[i, 0] == 0 else 0 for i in range(n)]
    return np.reshape(z, np.shape(a))


def NOR(a, b):
    n = np.shape(a)[0]
    z = [1 if a[i, 0] == 0 and b[i, 0] == 0 else 0 for i in range(n)]
    return np.reshape(z, np.shape(a))


def getXORTrain(num_bits):
    x_train = []
    y_train = []
    num_size = int(num_bits / 2)
    num_samples = 2 ** num_bits
    for i in range(num_samples):
        x = convert(i, 2, num_bits)
        a = x[0:num_size, :]
        b = x[num_size:num_bits, :]
        y = XOR(a, b)
        x_train.append(x)
        y_train.append(y)
    return [x_train, y_train]


def getORTrain(num_bits):
    x_train = []
    y_train = []
    num_size = int(num_bits / 2)
    num_samples = 2 ** num_bits
    for i in range(num_samples):
        x = convert(i, 2, num_bits)
        a = x[0:num_size, :]
        b = x[num_size:num_bits, :]
        y = OR(a, b)
        x_train.append(x)
        y_train.append(y)
    return [x_train, y_train]


def getANDTrain(num_bits):
    x_train = []
    y_train = []
    num_size = int(num_bits / 2)
    num_samples = 2 ** num_bits
    for i in range(num_samples):
        x = convert(i, 2, num_bits)
        a = x[0:num_size, :]
        b = x[num_size:num_bits, :]
        y = AND(a, b)
        x_train.append(x)
        y_train.append(y)
    return [x_train, y_train]


def getNANDTrain(num_bits):
    x_train = []
    y_train = []
    num_size = int(num_bits / 2)
    num_samples = 2 ** num_bits
    for i in range(num_samples):
        x = convert(i, 2, num_bits)
        a = x[0:num_size, :]
        b = x[num_size:num_bits, :]
        y = NAND(a, b)
        x_train.append(x)
        y_train.append(y)
    return [x_train, y_train]


def getNORTrain(num_bits):
    x_train = []
    y_train = []
    num_size = int(num_bits / 2)
    num_samples = 2 ** num_bits
    for i in range(num_samples):
        x = convert(i, 2, num_bits)
        a = x[0:num_size, :]
        b = x[num_size:num_bits, :]
        y = NOR(a, b)
        x_train.append(x)
        y_train.append(y)
    return [x_train, y_train]


def getTrains(num_bits, names):
    x_trains = []
    y_trains = []
    for name in names:
        if name == "OR":
            train = getORTrain(num_bits)
        elif name == "XOR":
            train = getXORTrain(num_bits)
        elif name == "AND":
            train = getANDTrain(num_bits)
        elif name == "NAND":
            train = getNANDTrain(num_bits)
        elif name == "NOR":
            train = getNORTrain(num_bits)
        else:
            train = [[], []]
        x_trains.append(train[0])
        y_trains.append(train[1])
    return [x_trains, y_trains]
