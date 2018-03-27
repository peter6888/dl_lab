'''
eExperienments on Tensorflow's rnn cells
'''
import tensorflow as tf
import numpy as np

batch_size = 5
input_size = 10
hidden_size = 3
sequence_size = 4
input_data = tf.placeholder(tf.float32, [batch_size, input_size])
def test_grucell():
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)
    hidden_state = initial_state
    output_of_cell, hidden_state = cell(input_data, hidden_state)
    inputs = np.random.randn(batch_size, input_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run((output_of_cell, hidden_state), feed_dict={input_data:inputs}))


def test_lstmcell():
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    hidden_state = initial_state
    output_of_cell, hidden_state = lstm_cell(input_data, hidden_state)
    inputs = np.random.randn(batch_size, input_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o_cell, states_cell = sess.run((output_of_cell, hidden_state), feed_dict={input_data:inputs})
        print("output of LSTM----")
        print(o_cell)
        print("memory cell of LSTM----")
        print(states_cell[0])
        print("hidden cell of LSTM----")
        print(states_cell[1])
        print("expect (True) output equals hidden cell:")
        print(np.array_equal(o_cell, states_cell[1]))

def test_bidirectional_rnn():
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    (outputs_fw, outputs_bw), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, np.random.randn(batch_size, sequence_size, input_size), \
                                                   initial_state_fw=cell_fw.zero_state(batch_size, dtype=tf.float64), \
                                                   initial_state_bw=cell_bw.zero_state(batch_size, dtype=tf.float64))
    print("outputs_fw.shape:{}, fw_state.shape:{}, bw_state.shape:{}".format(outputs_fw.get_shape(), fw_state.get_shape(), bw_state.get_shape()))
    state_concat = tf.concat([fw_state, bw_state], axis=1)
    print("state_concat.shape:{}".format(state_concat.get_shape()))
    print("shape after sum over sequence:{}".format(tf.reduce_sum(outputs_bw, axis=1)))

def test_lstmcell_concated():
    '''
    LSTM cell when hidden state concated
    '''
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=False, reuse=True)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    hidden_state = initial_state
    output_of_cell, hidden_state = lstm_cell(input_data, hidden_state)
    inputs = np.random.randn(batch_size, input_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o_cell, states_cell = sess.run((output_of_cell, hidden_state), feed_dict={input_data:inputs})
        print("output of LSTM----")
        print(o_cell)
        print("states cell of LSTM----")
        print(states_cell)

print("----gru cell output-----")
test_grucell()
#sample output
'''
2018-02-27 11:04:34.659133: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
(array([[-0.322102  ,  0.26052436,  0.03516981],
       [ 0.0235789 ,  0.13357721, -0.01362107],
       [ 0.01783257, -0.3178206 , -0.27473962],
       [-0.05341873,  0.17133303,  0.21480843],
       [-0.03101489, -0.154895  ,  0.40087253]], dtype=float32), array([[-0.322102  ,  0.26052436,  0.03516981],
       [ 0.0235789 ,  0.13357721, -0.01362107],
       [ 0.01783257, -0.3178206 , -0.27473962],
       [-0.05341873,  0.17133303,  0.21480843],
       [-0.03101489, -0.154895  ,  0.40087253]], dtype=float32))
'''
print("----lstm cell output-----")
test_lstmcell()
'''
----lstm cell output-----
output of LSTM----
[[ 0.12630886  0.05107639  0.30108988]
 [ 0.14326814  0.10635206  0.18994492]
 [ 0.119052   -0.03605069  0.3540535 ]
 [ 0.09831928  0.1978422  -0.17794694]
 [ 0.19544512  0.0061849   0.01266448]]
memory cell of LSTM----
[[ 0.26578057  0.13440524  0.74452716]
 [ 0.41908556  0.22692595  0.40551957]
 [ 0.20927444 -0.08193653  0.6071308 ]
 [ 0.1547601   0.370478   -0.32209384]
 [ 0.38525528  0.01277554  0.03883556]]
hidden cell of LSTM----
[[ 0.12630886  0.05107639  0.30108988]
 [ 0.14326814  0.10635206  0.18994492]
 [ 0.119052   -0.03605069  0.3540535 ]
 [ 0.09831928  0.1978422  -0.17794694]
 [ 0.19544512  0.0061849   0.01266448]]
'''

print("----lstm cell (hidden status no in tuple")
test_lstmcell_concated()
# sample output
'''
----lstm cell (hidden status no in tuple
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x111a2bb50>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
output of LSTM----
[[ 0.20295614 -0.22860827 -0.04467329]
 [ 0.00203407 -0.01209573  0.07957634]
 [ 0.17795433  0.22867016  0.14558344]
 [-0.11914554  0.03161938  0.20328507]
 [ 0.08281051  0.33082792  0.12093458]]
states cell of LSTM----
[[ 0.6540731  -0.605692   -0.08177677  0.20295614 -0.22860827 -0.04467329]
 [ 0.00499737 -0.03487571  0.2081982   0.00203407 -0.01209573  0.07957634]
 [ 0.5321151   0.60825545  0.2920621   0.17795433  0.22867016  0.14558344]
 [-0.1871059   0.07549004  0.5400781  -0.11914554  0.03161938  0.20328507]
 [ 0.18576448  0.4512465   0.2278376   0.08281051  0.33082792  0.12093458]]
'''

test_bidirectional_rnn()