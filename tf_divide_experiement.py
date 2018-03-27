import tensorflow as tf
def tf_div():
    batch_size = 4
    decoder_T = 2
    e = tf.random_normal(shape=[batch_size, decoder_T])
    # Equation (7)
    denominator = tf.reduce_sum(tf.exp(e[:, :-1]), axis=1, keep_dims=True)  # ignore the last e
    attn_score = tf.divide(tf.exp(e), denominator)  # shape (batch_size, decoder_T)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(attn_score)
        print(ret)

def hWh():
    # for equation (6)
    batch_size = 4
    hidden_state_size = 3
    decoder_T = 2
    decoder_state = tf.random_normal(shape=[batch_size, hidden_state_size])
    decoder_states = tf.random_normal(shape=[decoder_T, batch_size, hidden_state_size])
    W_d_attn = tf.random_normal(shape=[hidden_state_size, hidden_state_size])

    # tf.einsum implementation
    # return shape [T, batch_size, hidden_state_size]
    decoder_states_dot_W = tf.einsum("ij,tbi->tbj", W_d_attn, decoder_states)
    # return shape [batch_size, T]
    e = tf.einsum("tbi,bi->bt", decoder_states_dot_W, decoder_state)

    # Equation (7)
    denominator = tf.reduce_sum(tf.exp(e[:, :-1]), axis=1, keep_dims=True)  # ignore the last e
    attn_score = tf.divide(tf.exp(e), denominator)  # shape (batch_size, decoder_T)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(attn_score)
        print(ret)
        print("shape of ret {}".format(ret.shape))
    '''
                    W_d_attn = tf.get_variable('W_d_attn', shape=(1, 1, decoder_hidden_vec_size, decoder_hidden_vec_size), \
                                           initializer=tf.contrib.layers.xavier_initializer())
                decoder_T = len(decoder_states)

                if decoder_T > 1:
                    # Equation (6)
                    decoder_states_ex = tf.expand_dims(decoder_states_stack, axis=2)
                    decoder_states_dot_W = nn_ops.conv2d(decoder_states_ex, W_d_attn, [1, 1, 1, 1],
                                        # (batch_size, decoder_T)
                    e = tf.einsum("ijkl,jl->ij", decoder_states_dot_W, decoder_state)
                    # Equation (7)
                    denominator = tf.reduce_sum(tf.exp(e[:, :-1]), axis=1, keep_dims=True)  # ignore the last e
                    attn_score = tf.divide(tf.exp(e), denominator)  # shape (batch_size, decoder_T)
    '''


if __name__ == "__main__":
    tf_div()
    print("hWh output--")
    hWh()