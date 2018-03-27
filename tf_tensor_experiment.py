import tensorflow as tf

def tf_vector_size_to_half():
    batch_size = 4
    vector_size = 6
    old_vectors = tf.random_normal(shape=[batch_size, vector_size])
    eyes = tf.concat([tf.eye(vector_size//2) * 0.5, tf.eye(vector_size//2) * 0.5], axis=0)
    new_vectors = tf.matmul(old_vectors, eyes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _old, _new = sess.run([old_vectors, new_vectors])
        print("old:")
        print(_old)
        print("new:")
        print(_new)

if __name__ == "__main__":
    tf_vector_size_to_half()