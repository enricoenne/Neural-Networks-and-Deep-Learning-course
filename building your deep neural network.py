import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
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

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # derivative of cost with respect to AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1- AL))

    current_cache = caches[-1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l+1)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA'+l+2], current_cache,'relu')

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return(parameters)



if __name__ == '__main__':
    parameters = initialize_parameters_deep([5, 4, 3])

    print(parameters)