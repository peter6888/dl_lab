'''
Test RNNs
'''
import tensorflow as tf
import numpy as np
batch_size = 5
input_size = 10
seq_size = 20
hidden_size = 3
input_data = tf.placeholder(tf.float32, [batch_size, seq_size, input_size])

def test_dynamic_gru():
    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=input_data, \
                                      initial_state=gru_cell.zero_state(tf.shape(input_data)[0], tf.float32), \
                                      dtype=tf.float32)
    feed_dict = {input_data:np.random.randn(batch_size, seq_size, input_size)}
    print("input.shape {},{},{}".format(batch_size, seq_size, input_size))
    print("hidden layer size {}".format(hidden_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o, s = sess.run([outputs, state], feed_dict=feed_dict)
        print("outputs.shape {}, state.shape {}".format(o.shape, s.shape))

def test_tf_bool():
    bool_data = tf.placeholder(tf.bool)
    # if true 20, else -10
    result = tf.cond(bool_data, lambda:20, lambda:-10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = result.eval(feed_dict={bool_data: True}, session=sess)
        print(r)

def test_tf_n_bool():
    bool_data = tf.placeholder(tf.bool, shape = [input_size,1 ])
    dw = tf.placeholder(tf.float32, shape = [input_size,1])
    # if true 20, else -10
    condition_assign_op = tf.where(bool_data, dw*dw, tf.zeros_like(dw))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_bool = np.random.randn(input_size, 1) > 0.5
        f_dict = {bool_data: input_bool}
        f_dict[dw] = np.random.randn(input_size, 1)
        r = sess.run(condition_assign_op, feed_dict=f_dict)
        print(r)
#sample output
'''
[[0.6925733]
 [0.       ]
 [0.       ]
 [0.       ]
 [0.       ]
 [0.       ]
 [0.       ]
 [0.       ]
 [1.1562108]
 [0.       ]]
'''

test_dynamic_gru()
test_tf_bool()
test_tf_n_bool()