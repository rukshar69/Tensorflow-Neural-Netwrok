# the number of hidden layer nodes is somewhere
# between the number of input layers and the number
# of output layers.

#64 nodes to cover the 64 pixels in the image.
# 10 output layer nodes to predict the digits.

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
#plt.gray()
#plt.matshow(digits.images[1])
#plt.show()

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

#print(X[0,:])

from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

import numpy as np
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)


import numpy as np
nn_structure = [64, 30, 10]
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

#first step is to initialise the weights for each layer.

# To make it easy to organise the various layers,
# we’ll use Python dictionary objects (initialised by {}).
#  Finally, the weights have to be initialised with random
#  values – this is to ensure that the neural network will
#  converge correctly during training.
import numpy.random as r
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

#The next step is to set the mean accumulation values
# ΔW and Δb to zero (they need to be the same size as the
# weight and bias matrices):

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn_SGD(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%50 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values,
            # to be used in the gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] = np.dot(delta[l+1][:,np.newaxis],
                                       np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] = delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (tri_W[l] + lamb * W[l])
                b[l] += -alpha * (tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func



W, b, avg_cost_func = train_nn_SGD(nn_structure, X_train, y_v_train)

print(len(avg_cost_func))

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()
#our average cost function value has started to “plateau” and
# so further increases in the number of iterations isn’t likely to improve
# the performance of the network by much.

def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
print(accuracy_score(y_test, y_pred)*100)

#and the averaging over m samples removed):
#very easy transition from batch to stochastic gradient descent.
# The stochastic component is in the selection of the random
# selection of training sample.  However, if we use the
# scikit-learn test_train_split function the random selection
# has already occurred, so we can simply iterate through each
# training sample, which has a randomised order

#batch gradient descent after 1000 iterations outperforms SGD.
#   On the MNIST test set, the SGD run has an accuracy of 94%
# compared to a BGD accuracy of 96%.  WHY?
#SGD is noisy.it responds to the effects of each and every
# sample, and the samples  contain noisiness.
# While this can be a benefit in that it can act to
# “kick” the gradient descent out of local minimum values
# of the cost function, it can also hinder it settling down
#  into a good minimum.  SO
#  batch gradient descent has outperformed SGD
#There's a middle road Mini-batch





