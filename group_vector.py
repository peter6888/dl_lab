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
from group_vector_model import vector_size, batch_size
from group_vector_model import generate_data, vector_compare_concat, pred_op, loss_op, train_op, divide_data

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
    # Keep best model
    best_loss = 1e+10
    saver = tf.train.Saver()
    saved_model_path = "/tmp/model.ckpt"

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
                    # save best model
                    if best_loss > _loss:
                        best_loss = _loss
                        save_path = saver.save(sess, saved_model_path)
                        print("Model saved in path: {}".format(save_path))

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

