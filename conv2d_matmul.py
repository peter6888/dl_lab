import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np

batch_size = 5
attn_len = 3
attn_size = 2
vec_size = 4
def n_matmul():
    '''
    This function to prove that conv2d(h, W, [1,1,1,1], "SAME") can use for
        v = matmul(h, W)
    Where in batch scenarios
        v has shape (batch, v_length, vec_size)
        h has shape (batch, h_length, vec_size)
        W has shape (h_length, v_length)
    After expand W to shape (1, 1, h_length, v_length)
        and expand h to shape (batch_size, v_length, 1, vec_size)
        then call conv2d
    '''
    W_original = tf.constant(np.random.randn(attn_size, vec_size), dtype=tf.float32)
    h_original = tf.constant(np.random.randn(batch_size, attn_len, attn_size), dtype=tf.float32)
    W = tf.expand_dims(tf.expand_dims(W_original, 0), 0)
    h = tf.expand_dims(h_original, 2)
    v = nn_ops.conv2d(h, W, [1, 1, 1, 1], "SAME")

    W_reshape = tf.reshape(W, shape=(attn_size, vec_size))
    h_reshape = tf.reshape(h[-1,:,:,:], shape=(attn_len, attn_size))
    v_reshape = tf.matmul(h_reshape, W_reshape)

    h_dot_W = tf.einsum('ijk,kl->ijl', h_original, W_original)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, h_, W_ = sess.run([v,h, W])
        print(result[-1])
        print(result.shape)
        print("h shape:{}".format(h_.shape))
        print("W shape:{}".format(W_.shape))
        r_1, r_2 = sess.run([v_reshape, h_dot_W])
        print(r_1)
        print(r_1.shape)
        print("tf.einsum('ijk,kl->ijl')")
        print(r_2)


def attension_sum():
    encoded_states_expanded = np.random.randn(batch_size, attn_len, 1, attn_size)
    atten_vector = np.random.randn(batch_size, attn_len)
    context = np.einsum('ijkl,ij->ikl', encoded_states_expanded, atten_vector)
    # Equals below
    #att_vector_ex = np.expand_dims(np.expand_dims(atten_vector, axis=1), axis=1)

    #context_2 = np.sum(encoded_states_expanded * att_vector_ex, [2,3])
    #ValueError: operands could not be broadcast together with shapes (5,3,1,2) (5,1,1,3)

    #context_2 = np.sum(encoded_states_expanded * att_vector_ex, [2,3])
    print(context.shape)
    #print(context_2.shape)

def cnn_layer():
    input_width = 74
    input_height = 113
    batch_size = 3
    inputs = tf.Variable(tf.random_normal([batch_size, input_width, input_height, 1]))
    out = tf.layers.conv2d(inputs, filters=5, kernel_size=[8,8], strides=(64,64), padding="SAME", activation=tf.nn.relu)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out)
        print(o.shape)

if __name__ == "__main__":
    n_matmul()
    attension_sum()
    cnn_layer()

