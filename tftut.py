import tensorflow as tf
hello = tf.constant('hello there amy')
sess = tf.Session()
print(sess.run(hello))