'''
Experiment on different loss function
'''
import tensorflow as tf
import numpy as np

batch_size = 5
attn_len = 3
attn_size = 2
vec_size = 4
def tf_cosine():
    '''
    tf.losses.cosine_distance usage experiment
    Returns:
    '''
    v_batch = np.random.randn(batch_size, vec_size)
    u_batch = np.random.randn(batch_size, vec_size)
    loss = tf.losses.cosine_distance(v_batch, u_batch, dim=0)
    with tf.Session() as sess:
        loss = sess.run(loss)
        print(u_batch)
        print(v_batch)
        print(loss) #output a scalar!

def tf_batch_cosine():
    v_batch = np.random.randn(batch_size, vec_size)
    u_batch = np.random.randn(batch_size, vec_size)
    norm_v = tf.norm(v_batch, axis=1)
    norm_u = tf.norm(u_batch, axis=1)
    print(norm_v.get_shape())
    denominator = norm_v * norm_u
    print(denominator.get_shape())
    numerator = tf.einsum("ij,ij->i", tf.constant(v_batch), tf.constant(u_batch))#tf.matmul(v_batch, u_batch, transpose_b=True)
    print(numerator.get_shape())
    batched_cosine = tf.divide(numerator, denominator)
    print(batched_cosine.get_shape())
    with tf.Session() as sess:
        normed_v = sess.run(norm_v)
        print(normed_v)
        print("-----")
        print(v_batch)
    return

def tf_where():
    m = 1
    batched_cosine = tf.constant(np.random.randn(batch_size))
    z_flat = tf.constant([True] * batch_size)
    loss = tf.where(z_flat, batched_cosine, -batched_cosine)
    print(loss.get_shape())

if __name__ == "__main__":
    tf_cosine()
    tf_batch_cosine()
    print("----tf_where----")
    tf_where()