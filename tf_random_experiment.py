import tensorflow as tf

def tf_random_int():
    int_seq = tf.random_uniform(shape = [3, 2, 2], minval=1, maxval=4, dtype=tf.int32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _seq = sess.run(int_seq)
        print(_seq)

if __name__ == "__main__":
    tf_random_int()