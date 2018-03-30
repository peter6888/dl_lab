import tensorflow as tf
def test_onehot():
    oh = tf.one_hot([0,0,1], depth=2)
    return oh

def test_onehot_boolean():
    oh = tf.one_hot([True, False, True], depth=2)
    return oh

if __name__=="__main__":
    o_h = test_onehot()
    ob_h = test_onehot_boolean()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _o = sess.run(o_h)
        print(_o)
        ''' output
        [[1. 0.]
        [1. 0.]
        [0. 1.]]
        '''
        _ob = sess.run(ob_h)

        print("output shape {}".format(ob_h.get_shape()))
        print(_ob)
        ''' output
        [[0. 1.]
        [1. 0.]
        [0. 1.]]
        '''