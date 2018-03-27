import tensorflow as tf
import numpy as np

def tf_shape():
    inputs = tf.placeholder(dtype=tf.float32, shape=[2, None])
    #n1 = inputs.get_shape().as_list()[1]
    '''
    variable_scope.py", line 763, in _get_single_variable
    "but instead was %s." % (name, shape))
ValueError: Shape of a new variable (W) must be fully defined, but instead was (?, 5).
    '''

    n1 = tf.shape(inputs).as_list()[1]
    W = tf.get_variable("W", shape=[n1, 5], dtype=tf.float32, initializer=tf.zeros_initializer())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_out = sess.run(W, feed_dict={inputs: np.random.randn(2,3)})
        print(w_out)

if __name__ == "__main__":
    tf_shape()