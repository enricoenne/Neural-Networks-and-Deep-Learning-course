import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for i in range(1,L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters['b' + str(i)] = np.random.randn(layer_dims[i], 1) * 0.01

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    # relu activation for all the hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.multipy(Y, np.log(AL)) + np.multiply(1-Y, np.lof(1 - AL)))
    cost = np.squeeze(cost)

    return cost

if __name__ == '__main__':
    parameters = initialize_parameters_deep([5, 4, 3])

    print(parameters)