#http://adventuresinmachinelearning.com/neural-networks-tutorial/
import numpy as np
w1 = np.array([[0.2, 0.2, 0.2], #1st neuron of layers 2
               [0.4, 0.4, 0.4], #2nd neuron of layers 2
               [0.6, 0.6, 0.6]])#3rd neuron of layers 2
#the colums represent 3 input neuron
w2 = np.zeros((1, 3))
w2[0,:]= np.array([0.5, 0.5, 0.5])
#needed to match dimension with W1

#print(w2)
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

def f(x):
    return 1 / (1 + np.exp(-x))

#feed forward
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        #Setup the input array which the weights will be multiplied
        # by for each layer
        #If it's the first layer, the input array will be the x input vector
        #If it's not the first layer, the input to the next layer will be the
        #output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((w[l].shape[0],))
        #loop through the rows of the weight array
        for i in range(w[l].shape[0]):
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            #add the bias
            f_sum += b[l][i]
            #finally use the activation function to calculate the
            #i-th output i.e. h1, h2, h3
            h[i] = f(f_sum)
            print(h)
    return h

w = [w1, w2]
b = [b1, b2]
#a dummy x input vector
x = [1.5, 2.0, 3.0]

#hWB =simple_looped_nn_calc(3, x, w, b)
#print('final H ',hWB)

#increased efficiency with matrix multiplication
def matrix_feed_forward_calc(n_layers, x, w, b):


    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h

h_matrix = matrix_feed_forward_calc(3, x, w, b)