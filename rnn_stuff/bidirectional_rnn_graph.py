import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

from matplotlib import pyplot as plt
from collections import Counter


def plotGraph(x,y,title,xAxisTitle,yAxisTitle):
    plt.scatter(x,y)
    plt.title(title)
    plt.ylabel(yAxisTitle)
    plt.xlabel(xAxisTitle)
    plt.show()

def plotHistogram(array,title,xAxisTitle,yAxisTitle):
    plt.hist(array)
    plt.title(title)
    plt.ylabel(yAxisTitle)
    plt.xlabel(xAxisTitle)
    plt.show()

def weightVariable(x):
    a = x.ravel()
    return a


def weightCount(x):
    a = x.ravel()
    myRoundedList = [ round(elem,1) for elem in a]
    d = Counter(myRoundedList)
    return d

hm_epochs = 10 #10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 256 #256
rate_parameter = 4


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Hidden layer weights => 2*n_hidden because of forward + backward cells
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
layer = {'weights':tf.Variable(tf.random_poisson(rate_parameter,[2*rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_poisson(rate_parameter,[n_classes]))}

x_in = tf.transpose(x, [1,0,2])
x_in = tf.reshape(x_in, [-1, chunk_size])
x_in = tf.split(x_in, n_chunks, 0)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Define lstm cells with tensorflow
# Forward direction cell
lstm_fw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0,state_is_tuple=True)
# Backward direction cell
lstm_bw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0,state_is_tuple=True)


#lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
try:
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_in,
                                                 dtype=tf.float32)
except Exception:  # Old TensorFlow version only returns outputs not states
    outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_in,
                                           dtype=tf.float32)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']




prediction = output
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels= y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    weights = weightVariable(sess.run(layer["weights"]));
    weightCounter = weightCount(sess.run(layer["weights"]));

    plotHistogram(weights, "Weight Distribution of mnist RNN",
                  "Value of Weights", "Number of Edges")

    plotGraph(weightCounter.keys(), weightCounter.values(), "Weight Distribution of mnist RNN",
              "Value of Weights", "Number of Edges")

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    sum_accuracy = 0

    for i in range(int(mnist.test.num_examples / 100)):
        epoch_x, epoch_y = mnist.test.next_batch(100)
        epoch_x = epoch_x.reshape((100, n_chunks, chunk_size))
        sum_accuracy += accuracy.eval({x: epoch_x, y: epoch_y})

    print('Accuracy:', sum_accuracy)

    weights = weightVariable(sess.run(layer["weights"]));
    weightCounter = weightCount(sess.run(layer["weights"]));

    plotHistogram(weights, "Weight Distribution of mnist RNN",
              "Value of Weights", "Number of Edges")

    plotGraph(weightCounter.keys(),weightCounter.values(), "Weight Distribution of mnist RNN",
                  "Value of Weights", "Number of Edges")


