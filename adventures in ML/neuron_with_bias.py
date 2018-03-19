import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)
w = 5.0
b1 = -8.0
b2 = 0.0
b3 = 8.0
l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'
for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x*w+b)))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_wb(x)')
plt.legend(loc=2)
plt.show()

#the w1 has been increased to simulate a more defined “turn on” function.
# by varying the bias “weight” b, you can change when the
#  node activates.Therefore, by adding a bias term,
# you can make the node simulate  if function, i.e.
# if (x > z) then 1 else 0.
# Without a bias term, you are unable to vary the z
# in that if statement,
# it will be always stuck around 0.
# This is obviously very useful if you are
# trying to simulate conditional relationships.