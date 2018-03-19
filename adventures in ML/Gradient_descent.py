#The gradient  gives directional information –
# if it is positive with respect to an increase in w,
# a step in that direction will lead to an increase in the error.
# If it is negative with respect to an increase in w
# a step in that will lead to a decrease in the error.
# Obviously, we wish to make a step in w that will lead to a
# decrease in the error. The magnitude of the gradient
#
# the “steepness” of the slope,
# gives an indication of how fast the error curve
#  is changing at that point.
# The higher the magnitude of the gradient,
#  the faster the error is changing at that point with
# respect to w.

#The gradient descent method uses the gradient to make an
# informed step change in w to lead it towards the minimum
# of the error curve.
# This is an iterative method, that involves multiple steps.

#the step size α will determine how quickly the solution converges
# on the minimum error. However, this parameter has to be tuned
# – if it is too large, you can imagine the solution bouncing
# around on either side of the minimum
#  This will result in an optimisation of w that does not converge
# . As this iterative algorithm approaches the minimum,
# the gradient or change in the error with each step will reduce.
#
#  that the gradient lines will “flatten out” as the solution point
#  approaches the minimum. As the solution approaches the minimum
# error, because of the decreasing gradient, it will result in only
#  small improvements to the error.  When the solution approaches
# this “flattening” out of the error we want to exit the iterative
# process.  This exit can be performed by either stopping after
# a certain number of iterations or via some sort of
# “stop condition”.  This stop condition might be when the change
# in the error drops below a certain limit, often called the
# precision.

x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

def df(x):
    y = 4 * x**3 - 9 * x**2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

print("The local minimum occurs at %f" % x_new)