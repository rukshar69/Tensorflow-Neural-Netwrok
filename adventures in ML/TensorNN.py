from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])


#Notice the x input layer is 784 nodes corresponding to the 28 x 28 (=784) pixels,
# and the y output layer is 10 nodes corresponding to the 10 possible digits.
#  Again, the size of x is (? x 784), where the ? stands for an as yet unspecified number of
# samples to be input – this is the function of the placeholder variable.

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')
#a random normal distribution with a mean of zero and a standard deviation of 0.03.

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

#the softmax function, or normalized exponential function is a generalization of
# logistic function that "squashes" a K-dimensional vector
#  of arbitrary real values to a K-dimensional vector  of real values in the range (0, 1)
# that add up to 1

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
#the cross entropy cost function,
#The first is the summation of the logarithmic products
# and additions across all the output nodes.
# The second is taking a mean of this summation across all the training samples.

#The first line is an operation converting the output y_ to a clipped version,
# limited between 1e-10 to 0.999999.  This is to make sure that we never get a
# case were we have a log(0) operation occurring during training – this would return NaN
# and break the training process.
# The second line is the cross entropy calculation.

#tf.reduce_sum function – this function basically takes the sum of a given axis
# of the tensor you supply.  In this case, the tensor that is supplied is the
#  element-wise cross-entropy calculation for a single node

#y and y_clipped in the above calculation are (m x 10) tensors
#  we need to perform the first sum over the 2nd axis.
#  This is specified using the axis=1 argument,
#  where “1”  refers to the second axis

#we have an (m x 1) tensor.
#  To take the mean of this tensor and complete our cross entropy cost calculation

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#correct_prediction  use of the TensorFlow tf.equal  which returns True or False
# depending on whether to arguments supplied to it are equal.
#  The tf.argmax function returns the index of the maximum value in a vector / tensor.
#   Therefore, the correct_prediction  returns a tensor of size (m x 1)
#  of True and False values designating whether the neural network has correctly
#  predicted the digit.
# We then want to calculate the mean accuracy from this tensor
#  – first we have to cast the type of the correct_prediction operation from
# a Boolean to a TensorFlow float in order to perform the reduce_mean operation.

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

