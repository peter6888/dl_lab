import tensorflow as tf
def test_onehot():
    oh = tf.one_hot([0,0,1], depth=2)
    return oh

def test_onehot_boolean():
    oh = tf.one_hot([True, False, True], depth=2)
    return oh

def test_expand_dims():
    before_expand = tf.random_normal(shape=[5])
    after_expand = tf.expand_dims(before_expand, axis=1)
    return before_expand, after_expand
''' sample output 
Before expand shape=(5,), values are [-1.64261639  1.18676174  1.16362131  0.76668751  0.5099209 ]
After expand shape=(5, 1), values are [[-1.64261639]
 [ 1.18676174]
 [ 1.16362131]
 [ 0.76668751]
 [ 0.5099209 ]]
'''
if __name__=="__main__":
    o_h = test_onehot()
    ob_h = test_onehot_boolean()
    b, a = test_expand_dims()
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
        _b, _a = sess.run([b, a])
        print("Before expand shape={}, values are {}".format(_b.shape, _b))
        print("After expand shape={}, values are {}".format(_a.shape, _a))
