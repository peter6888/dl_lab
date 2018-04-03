import tensorflow as tf
import numpy as np

def np_argmax():
    '''
    Experiment on np.argmax string use .astype(np.int32)
    Returns:None
    '''
    labels = np.array(['0', '1', '0'])
    labels = labels.astype(np.int32)
    pred = np.array([[0.179,0.0],[0.752,0.0],[0.515,0.0]])
    count = np.sum(labels==np.argmax(pred, axis=1))
    print("accuraccy {}/{}={}".format(count, len(labels), count/len(labels)))

def np_argmax_string():
    '''
    Experiment on np.argmax string, which the label is actually a string
    Returns: None
    '''
    labels = np.array(['0', '1', '0'])
    pred = np.array([[0.179,0.0],[0.752,0.0],[0.515,0.0]])
    indices = np.argmax(pred, axis=1)
    indices_str = [str(_) for _ in indices]
    count = np.sum(labels==indices_str)
    print("accuraccy {}/{}={}".format(count, len(labels), count/len(labels)))

def tf_argmax():
    '''
    Experiment on tf.argmax
    Returns: None
    '''
    z = tf.constant([0,1,0])
    pred = tf.constant([[0.179,0.0],[0.752,0.0],[0.515,0.0]])
    pred_softmax = tf.nn.softmax(pred)
    pred_index = tf.argmax(pred_softmax, axis=1, output_type=tf.int32)
    # Below comparation try to elementwise compare the indices on z and pred_index, but will going to always get 0
    #compare = np.sum(z==pred_index)
    # use tf.equal and tf.cast
    elements_equal = tf.equal(z, pred_index)
    elements_equal_as_ints = tf.cast(elements_equal, tf.int32)
    count = tf.reduce_sum(elements_equal_as_ints)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _z, _pred, _pred_softmax, _pred_index, _elements_equal, _elements_equal_as_ints, _count = \
            sess.run([z, pred, pred_softmax, pred_index, elements_equal, elements_equal_as_ints, count])
        print("z----")
        print(_z)
        print("pred----")
        print(_pred)
        print("pred_softmax----")
        print(_pred_softmax)
        print("pred_index----")
        print(_pred_index)
        #print("compare----")
        #print(compare)
        print("elements_equal--")
        print(_elements_equal)
        print("elements_equal_as_ints--")
        print(_elements_equal_as_ints)
        print("count-----")
        print(_count)

''' Sample result
z----
[0 1 0]
pred----
[[0.179 0.   ]
 [0.752 0.   ]
 [0.515 0.   ]]
pred_softmax----
[[0.54463094 0.45536914]
 [0.6796143  0.32038566]
 [0.6259779  0.37402216]]
pred_index----
[0 0 0]
elements_equal--
[ True False  True]
elements_equal_as_ints--
[1 0 1]
count-----
2
'''

if __name__ == "__main__":
    tf_argmax()
    np_argmax_string()
    np_argmax()