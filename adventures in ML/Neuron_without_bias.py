
import matplotlib.pylab as plt
import numpy as np
w1 = 0.5
w2 = 1.0
w3 = 2.0
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'
x = np.arange(-8, 8, 0.1)
for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x*w))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)
plt.show()

#useful if we want to model different strengths of
# relationships between the input and output variables
#However,  if we only want  output to change when x is greater than 1?
# This is where the bias comes in
