import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from collections import Counter


def plotGraph(x,y,title,xAxisTitle,yAxisTitle):
    plt.scatter(x,y)
    plt.title(title)
    plt.ylabel(yAxisTitle)
    plt.xlabel(xAxisTitle)
    plt.show()


def weightCount(x):
    a = x.ravel()
    myRoundedList = [ round(elem,2) for elem in a]
    d = Counter(myRoundedList)
    return d

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128
learning_rate = 0.00001
rate_parameter = 4

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 1.0
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



weights = {'W_conv1': tf.Variable(tf.random_poisson(rate_parameter,[5, 5, 1, 32])),
            'W_conv2': tf.Variable(tf.random_poisson(rate_parameter,[5, 5, 32, 64])),
            'W_fc': tf.Variable(tf.random_poisson(rate_parameter,[7 * 7 * 64, 1024])),
            'out': tf.Variable(tf.random_poisson(rate_parameter,[1024, n_classes]))}

biases = {'b_conv1': tf.Variable(tf.random_poisson(rate_parameter,[32])),
          'b_conv2': tf.Variable(tf.random_poisson(rate_parameter,[64])),
          'b_fc': tf.Variable(tf.random_poisson(rate_parameter,[1024])),
          'out': tf.Variable(tf.random_poisson(rate_parameter,[n_classes]))}

xUse = tf.reshape(x, shape=[-1, 28, 28, 1])

conv1 = tf.nn.relu(conv2d(xUse, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
fc = tf.nn.dropout(fc, keep_rate)

output = tf.matmul(fc, weights['out']) + biases['out']



prediction = output
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 2
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            ##print("Running",_,"batch of",epoch)
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    sum_accuracy = 0

    for i in range(int(mnist.test.num_examples / 100)):
        epoch_x, epoch_y = mnist.test.next_batch(100)
        sum_accuracy+=accuracy.eval({x: epoch_x, y: epoch_y})

    print('Accuracy:',sum_accuracy)

    weightLayer1 = weightCount(sess.run(weights["W_conv1"]));
    weightLayer2 = weightCount(sess.run(weights["W_conv2"]));
    weightFullyConnectedLayer = weightCount(sess.run(weights["W_fc"]));
    weightOutputLayer = weightCount(sess.run(weights["out"]));

    weightTotalCounter = weightLayer1+weightLayer2+weightFullyConnectedLayer+weightOutputLayer;

    plotGraph(weightTotalCounter.keys(), weightTotalCounter.values(), "Weight Distribution of mnist CNN", "Value of Weights","Number of Edges")

