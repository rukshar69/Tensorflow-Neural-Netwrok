import  tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1*x2
#computational node

print(result) #weird answer because it's a node not a number


with tf.Session() as sess: #you don't need to call close() anymore
    print(sess.run(result))