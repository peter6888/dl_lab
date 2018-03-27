'''
This is an example model to use tf.layers.dense to group vectors
Data have two group of vectors.
    1) all vectors have same size, say 3
    2) group A vectors have larger values on one index, say 1
    3) group B vectors have larger values on another index, say 2
    4) The other index are relative small random data
    Examples:
        [100, 0.1, 0.1] in group A
        [0.1, 100, 0.1] in group B

Task, given two vectors, your model could tell if they are in same group
'''

import tensorflow as tf
import numpy as np

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

if __name__ == "__main__":
    l = generate_data(1000)
    dict_l = divide_data(l)
    print(len(dict_l["train"]),len(dict_l["val"]), len(dict_l["test"]))

    tensor_a = tf.placeholder(shape=[None, vector_size], dtype=tf.float32)
    tensor_b = tf.placeholder(shape=[None, vector_size], dtype=tf.float32)
    tensor_c = tf.placeholder(shape=[None], dtype=tf.int32)
    pred = pred_op(vector_compare_concat(tensor_a, tensor_b))
    loss = loss_op(pred, tensor_c)
    train = train_op(loss)
    train_data = dict_l["train"]
    val_data = dict_l["val"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            np.random.shuffle(train_data)
            a, b, c = zip(* train_data)
            for j in range(len(a)//batch_size):
                _pred, _loss, _ = sess.run([pred, loss, train], feed_dict={tensor_a: a[j*batch_size:(j+1)*batch_size], tensor_b: b[j*batch_size:(j+1)*batch_size], tensor_c: c[j*batch_size:(j+1)*batch_size]})
            #print(_pred.shape) # output shape [batch_size, 2] # binary classification
            #print(_pred)
            if i%1000 == 0:
                print("Cross-Entropy loss:{}".format(_loss))
                if i > 5000:
                    np.random.shuffle(val_data)
                    val_a, val_b, val_y = zip(*val_data)
                    _c = sess.run(tf.nn.softmax(pred), feed_dict={tensor_a: val_a, tensor_b: val_b})
                    y_hat = np.argmax(_c, axis=1)
                    print(np.sum(y_hat==np.array(val_y)) / len(val_y))
                    #print(y_hat.shape)
                    #print(len(val_y))
'''
Cross-Entropy loss:441.1654968261719
Cross-Entropy loss:0.30894121527671814
Cross-Entropy loss:0.24041341245174408
Cross-Entropy loss:0.21878936886787415
Cross-Entropy loss:0.20873117446899414
Cross-Entropy loss:0.20419609546661377
Cross-Entropy loss:0.20192311704158783
173
Cross-Entropy loss:0.20131425559520721
173
Cross-Entropy loss:0.20093075931072235
173
Cross-Entropy loss:0.20085400342941284
173
'''

