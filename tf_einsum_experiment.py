import tensorflow as tf
from tensorflow.python.ops import math_ops
def rank3_mul():
    '''
    Test tf.einsum with rank 3 Tensor
    '''
    batch_size = 2
    seq_length = 3
    hidden_vector_size = 4

    rank2state = tf.random_normal(shape=[batch_size, hidden_vector_size])
    rank3states = tf.random_normal(shape=[batch_size, seq_length, hidden_vector_size])
    ''' Error information for below line
Traceback (most recent call last):
  File "tf_einsum_experiment.py", line 20, in <module>
    rank3_mul()
  File "tf_einsum_experiment.py", line 12, in rank3_mul
    result = tf.einsum("bi,bTi->bi", rank2state, rank3states)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/special_math_ops.py", line 210, in einsum
    axes_to_sum)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/special_math_ops.py", line 256, in _einsum_reduction
    raise ValueError()
ValueError
    '''
    #result = tf.einsum("bi,bTi->bT", rank2state, rank3states)
    '''
    math_ops.reduce_sum(decoder_state * encoder_states_dot_W, [2, 3])
    '''
    rank2state = tf.expand_dims(rank2state, axis=1)
    result = math_ops.reduce_sum(rank2state * rank3states, axis=-1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _r2, _r3, _result = sess.run([rank2state, rank3states, result])
        print("r2")
        print(_r2)
        print("r3")
        print(_r3)
        print("result")
        print(_result)

def rank2and3_to_rank3():
    batch_size = 2
    seq_length = 3
    hidden_vector_size = 4

    rank2matrix = tf.random_normal(shape=[hidden_vector_size, hidden_vector_size])
    rank3states = tf.random_normal(shape=[batch_size, seq_length, hidden_vector_size])

    #encoder_states_dot_W = np.einsum("ij,bTj->bTi", _W, _states)
    tf_result = tf.einsum("ij,btj->btj", rank2matrix, rank3states)
    print(tf_result.get_shape())
    rank2matrix_expanded = tf.expand_dims(tf.expand_dims(rank2matrix, axis=0), axis=0) #new shape [1, 1, hidden_vector_size, hidden_vector_size]
    rank3states_expanded = tf.expand_dims(rank3states, axis=2) #new shape [batch_size, seq_length, 1, hidden_vector_size]
    result = tf.reduce_sum(rank2matrix_expanded * rank3states_expanded, axis=2) #output shape [batch_size, seq_length, hidden_vector_size]
    print("result.shape")
    print(result.get_shape())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _tfresult, _result, _mat, _states = sess.run([tf_result, result, rank2matrix, rank3states])
        print("matrix:")
        print(_mat)
        print("states:")
        print(_states)
        print("result:")
        print(_result)
        print("result fro tf:")
        print(_tfresult)



if __name__ == "__main__":
    rank3_mul()
    rank2and3_to_rank3()