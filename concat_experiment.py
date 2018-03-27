'''
Experiments for tf.concat
'''
import tensorflow as tf
import numpy as np

def concat_1d():
    print("concat 1d experiment")
    a = tf.constant([1,2])
    b = tf.constant([2,3,5])
    c = tf.concat([a,b], axis=0) #need use axis=0
    with tf.Session() as sess:
        np_a, np_b, np_c = sess.run([a,b,c])
        print("input \t\na {}\nb {}\noutput c {}".format(np_a, np_b, np_c))

def concat_2d():
    print("concat 2d experiment")
    a = tf.constant([[1,2],[3,4]])
    b = tf.constant([[3,4,5], [5,6,7]])
    c = tf.concat([a,b], axis=1) #need use axis = 1
    with tf.Session() as sess:
        np_a, np_b, np_c = sess.run([a,b,c])
        print("input \t\na {}\nb {}\noutput c {}".format(np_a, np_b, np_c))

def placeholder_ex():
    states = tf.placeholder(shape=(None, None, 5), dtype=tf.float32)
    input1 = np.random.randn(1,3,5)
    input2 = np.random.randn(2,2,5)
    with tf.Session() as sess:
        print(sess.run(states, feed_dict={states:input1}))
        print(sess.run(states, feed_dict={states:input2}))

def concat_3tensors():
    t = 100
    batch_size = 16
    vector_size = 256
    vector2_size = 512
    decoder_outputs = tf.random_normal([t, batch_size, vector_size])
    input_contexts  = tf.random_normal([t, batch_size, vector2_size])
    reduced_input_contexts = _reduce_contexts(input_contexts)
    decoder_contexts = tf.random_normal([t, batch_size, vector_size])
    contexts = tf.concat(values=[decoder_outputs, reduced_input_contexts, decoder_contexts], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        con = sess.run(contexts)
        print(con[0,0,0], con.shape)

def _reduce_contexts(larger_size_contexts):
    """
    Reduce context from vector_size * 2 to vector_size
    :param larger_size_contexts: Tensor shape ([t, batch_size, vector_size * 2])
    :return:smaller_size_contexts: Tensor shape ([t, batch_size, vector_size])
    """
    vector_size = (larger_size_contexts.get_shape().as_list()[2]) / 2
    with tf.variable_scope('reduce_contexts'):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce = tf.get_variable('w_reduce', [vector_size * 2, vector_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
        bias_reduce = tf.get_variable('bias_reduce', [vector_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))

    # Apply linear layer
    reduced_contexts = tf.nn.relu(tf.einsum("ijk,kl->ijl", larger_size_contexts, w_reduce) + bias_reduce)

    return reduced_contexts

if __name__=="__main__":
    concat_1d()
    concat_2d()
    placeholder_ex()
    concat_3tensors()