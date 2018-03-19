import matplotlib.pylab as plt
import numpy as np
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

#activation function. In classification tasks  this activation function has to have a “switch on”
# characteristic – in other words, once the input is greater than a certain value, the output should change state
# i.e. from 0 to 1, from -1 to 1 or from 0 to >0.
# This simulates the “turning on” of a biological neuron.
# A common activation function that is used is the sigmoid function:
#the edge is “soft”,
# and the output doesn’t change instantaneously.
# so there is a derivative of the function
# it is important for the training algorithm