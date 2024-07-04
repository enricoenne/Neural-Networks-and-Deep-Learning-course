import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
from PIL import Image
from dnn_app_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def BandW_avg(img_array):
    return np.mean(img_array, axis=3)

def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

        cost = compute_cost(A2, Y)
        dA2 = -(np.divide(Y,A2) - np.divide(1-Y, 1-A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        costs.append(cost)
        if print_cost and i%500 == 0:
            print('cost at iteration {}: {}'.format(i, np.squeeze(cost)))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('learning rate = '+str(learning_rate))
    plt.show()

    return parameters

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    m = X.shape[1]

    parameters = initialize_parameters_deep(layer_dims)


    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        costs.append(cost)
        if print_cost and i%500 == 0:
            print('cost at iteration {}: {}'.format(i, np.squeeze(cost)))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('learning rate = '+str(learning_rate))
    plt.show()

    return parameters

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# a picture from the dataset
index = 10
plt.imshow(train_x_orig[index])
print('y = ' + str(train_y[0, index]))
plt.show()

# exploring dimensions of datasets, train and test
m_train = train_x_orig.shape[0]
print(m_train)
num_px = train_x_orig.shape[1]
print(train_x_orig.shape)
m_test = test_x_orig.shape[0]
print(m_test)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T


train_x = train_x_flatten / 255
test_x = test_x_flatten / 255


# two layer model
n_x = num_px * num_px * 3
n_h = 7
n_y = 1
layer_dims = (n_x, n_h, n_y)

parameters = two_layer_model(train_x, train_y, layer_dims = (n_x, n_h, n_y), num_iterations = 3000, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)

# index of wrong predictions
'''error_on_test = abs(test_y - predictions_test).nonzero()[1]

for i in error_on_test:
    plt.imshow(test_x_orig[i])
    plt.show()'''

# deep layer model
print('NOW DEEP MODEL')
layer_dims = [num_px*num_px*3, 30, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layer_dims = (n_x, n_h, n_y), num_iterations = 3000, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)

'''error_on_test = abs(test_y - predictions_test).nonzero()[1]

for i in error_on_test:
    plt.imshow(test_x_orig[i])
    plt.show()'''