import tensorflow as tf

def tf_attention():
    batch_size = 5
    max_total_time = 4
    input_vector_size = 3
    hidden_vector_size = 2
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_vector_size)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    inputs = tf.random_normal(shape=(batch_size, max_total_time, input_vector_size))
    #rnn_output[0] has shape [t, batch_size, hidden_vector_size]
    rnn_output = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state = initial_state)
    print("rnn_output[0] shape {}".format(rnn_output[0].get_shape()))

    # attention_states: [batch_size, max_time, hidden_vector_size(num_units)]
    attention_states = tf.transpose(rnn_output[0], [1, 0, 2])
    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(5, attention_states)

    decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_vector_size)
    attentioned_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism, attention_layer_size=5)
    '''
    #The `batch_size` argument passed to the `zero_state` method of this
    #     wrapper is equal to `true_batch_size * beam_width`.
    '''
    attention_initial_state = attentioned_decoder_cell.zero_state(batch_size * 4, tf.float32)

    attention_inputs = tf.random_normal(shape=[batch_size, hidden_vector_size])
    attentioned_result = attentioned_decoder_cell(attention_inputs, attention_initial_state)
    '''
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=num_units)
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _encoder_outputs, _state = sess.run(rnn_output)
        print("outputs---")
        print(_encoder_outputs)
        print("state-c-memory output-")
        print(_state[0])
        print("state-h-hidden state-")
        print(_state[1])
        print("attention")
        _attentioned_result = sess.run(attentioned_result)
        print(_attentioned_result)

if __name__ == "__main__":
    tf_attention()