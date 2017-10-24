import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#emulating dead neurons ->some dysfunctional neuron
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

'''
 The strides parameter dictates the movement of the window. In this case, we just move 1 pixel at a time for the conv2d function,
  and 2 at a time for the maxpool2d function. The ksize parameter is the size of the pooling window. In our case, we're choosing 
  a 2x2 pooling window for pooling.
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {# 5 x 5 convolution, 1 input image, 32 outputs
                'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # Reshape conv2 output to fit fully connected layer

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate) #"dropout." The idea of it is to mimic dead neurons in your own brain.
    #We're going to keep 80% of our neurons per training iteration

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)

    '''
    
    “logit layer” in the above graph) which uses cross entropy as a cost/loss function. 
    
    cross entropy:  the cross entropy between two probability distributions  p and q over 
    the same  set of events measures the average number of bits needed to identify an event drawn from the set
    
     In probability theory, the output of the softmax function can be used to represent a categorical distribution – 
     that is, a probability distribution over K different possible outcomes.
     
     For Tensorflow: logits  imply  this Tensor is the quantity that being mapped 
     to probabilities by the Softmax. 
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #Measures the probability error in discrete classification tasks in which the classes are
    # mutually exclusive (each entry is in exactly one class).
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #implement the adam algo -> gradient descent

    '''
    However, TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
     The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative 
     of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone.
      Consequently, TensorFlow can automatically produce derivatives. optimizers typically do this for you
    '''

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)