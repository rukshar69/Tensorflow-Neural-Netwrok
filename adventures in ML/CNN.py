# the MNIST dataset is  simple.
# The images are small (only 28 x 28 pixels),
#  are single layered (i.e. greyscale, rather than a coloured 3 layer RGB image)
# and include pretty simple shapes (digits only, no other objects).
#  Once we start trying to classify things in more complicated colour images,
# such as buses, cars, trains etc. , we run into problems with our accuracy.

#increase the number of layers in our neural network to make it deeper.
#   That will increase the complexity of the network
# and allow us to model more complicated functions.
# However, it will come at a cost – the number of parameters (i.e. weights and biases)
#  will rapidly increase.
# This makes the model more prone to overfitting and will prolong training times.
#   In fact, learning such difficult problems can become intractable for normal neural
#  networks.  This leads us to a solution – convolutional neural networks.

#Sparse connections – notice that not every input node is connected to the output nodes.
#
# Constant filter parameters / weights – each filter has constant parameters.
#  the filter moves around the image the same weights
#  Each filter therefore performs a  transformation across the whole image.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 50

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from
# mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

#he image input data is extracted using the mnist.train.nextbatch() ,
# which supplies a flattened 28×28=784 node, single channel greyscale  image.
#The format of the data to be supplied is [i, j, k, l]
#  i = the number of training samples,
# j = the height of the image,
# k = the weight
#  l =is the channel number.
#  Because we have a greyscale image, l will always be equal to 1
# (if we had an RGB image, it would be equal to 3).
# The MNIST images are 28 x 28, so both j and k are equal to 28.
# When we reshape the input data x into x_shaped,
#  theoretically we don’t know the size of the first dimension of x,
# so we don’t know what i is.  However, tf.reshape() allows us to put -1 in place of i
# and it will dynamically reshape based on the number of training samples
#  as the training is performed.  So we use [-1, 28, 28, 1] for
#  the second argument in tf.reshape()


#Tconvolution step is often called feature mapping.
# to classify well,  each convolutional stage  need multiple filters

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
# [1, 1, 1, 1]  strides .we want the filter to move in steps of 1 in both the x and y dir
    #  This information is conveyed in the strides[1] and strides[2] values –
    #  both equal to 1 in this case.
    # The first and last values of strides are always equal to 1,
    #  if they were not, we would be moving the filter between training samples
    # or between channels, which we don’t want to do.
    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]

    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

#Each channels is trained to detect features in the image.

#pooling
#Reduce the number of parameters  (called “down-sampling” for this reason)
#To make feature detection impervious to scale and orientation changes

# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

# we have to flatten out the output from the final convolutional layer.
#  It is now a 7×7 grid of nodes with 64 channels,
#  which equates to 3136 nodes per training sample.
#  We can use tf.reshape() to do what we need:
#dynamically calculated first dimension (the -1 above)=number of input samples in the training batch.

wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
#The function first takes the soft-max of the matrix multiplication,
# then compares it to the training target using cross-entropy.
#   The result is the cross-entropy calculation per training sample,

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

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
        test_acc = sess.run(accuracy,
                       feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

#Adam is Straightforward to implement.
#Computationally efficient.
#Little memory requirements.