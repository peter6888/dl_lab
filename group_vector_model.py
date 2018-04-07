import tensorflow as tf
import numpy as np
import os

lr = 0.003
vector_size = 3
batch_size = 200

def generate_data(counts, size=3):
    ret = []

    def get_a_b():
        large_value = 100.0
        a_index = 0
        b_index = 1
        np_a, np_b = np.ones(shape=size), np.ones(shape=size)
        np_a[a_index] = large_value
        np_b[b_index] = large_value
        return np.random.randn(size) * np_a, np.random.randn(size) * np_b

    for _ in range(counts):
        a1, b1 = get_a_b()
        a2, b2 = get_a_b()
        ret.append((a1, a2, 1))
        ret.append((b1, b2, 1))
        ret.append((a1, b1, 0))
        ret.append((a2, b2, 0))

    return ret

def vector_compare_concat(v_a, v_b):
    '''
    return the compare vector which defined by z(v_a,v_b) = tf.concat(v_a, v_b, v_a - v_b, v_a * v_b)
    Args:
        v_a: the left vector, shape [batch_size, vector_size]
        v_b: the right vector, shape [batch_size, vector_size]

    Returns:
        z_a_b: the compare vector, shape [batch_size, 4 * vector_size]
    '''
    z_a_b = tf.concat([v_a, v_b, v_a - v_b, v_a * v_b], axis=1)
    return z_a_b

def pred_op(v_z):
    '''
    Predict based on the concatenated compare vector
    Args:
        v_z: the concatenated compare vector, shape [batch_size, 4 * vector_size]

    Returns: Distribution possibility [batch_size, 2], a binary classification
    '''
    hidden_layer =  tf.layers.dense(inputs=v_z, units=2 * vector_size, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pred = tf.layers.dense(inputs=hidden_layer, units=2, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return pred

def loss_op(pred, y):
    '''
    Args:
        pred: the predicted distribution, shape [batch_size, vector_size]
        y: [batch_size, 2], one-hot vector for true label, [1,0] - false, [0,1] - true
    Returns: loss
    '''
    y_onehot = tf.one_hot(y, depth=2)
    loss = tf.losses.softmax_cross_entropy(y_onehot, pred)
    return loss

def train_op(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    _train_op = optimizer.minimize(loss)
    return _train_op

def divide_data(inputs):
    '''
    divide data to train, validation and test sets
    Args:
        inputs: list of vector pairs

    Returns: dictionary of lists for "train", "val" and "test". And 90%, 5%, 5% for each
    '''
    ret = {}
    np.random.shuffle(inputs)
    n = len(inputs)
    ret["train"] = inputs[:int(0.9 * n)]
    ret["val"]   = inputs[int(0.9 * n):int(0.95 * n)]
    ret["test"]  = inputs[int(0.95 * n):]

    return ret