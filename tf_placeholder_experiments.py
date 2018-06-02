import tensorflow as tf
import numpy as np

def tf_placeholder_none_experiment():
    print("tf_placeholder_none_experiment")
    shape_none_by_fixed_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _r = sess.run(shape_none_by_fixed_placeholder, feed_dict={shape_none_by_fixed_placeholder:np.array([[1,3,4],[3,4,5]])})
        print(_r)

if __name__=="__main__":
    tf_placeholder_none_experiment()