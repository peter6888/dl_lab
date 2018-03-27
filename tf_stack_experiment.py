import tensorflow as tf
import numpy as np

def tfstack_from_tensorlist():
    batch_size = 16
    decoder_hidden_size = 512
    encoder_hidden_size = 256
    decoder_t = 6
    vsize = 50000

    tensorlist = []
    attn_score1 = np.random.randn(batch_size, decoder_t)
    attn_score1 = tf.convert_to_tensor(attn_score1, np.float32)
    attn_score2 = np.random.randn(batch_size, decoder_t)
    attn_score2 = tf.convert_to_tensor(attn_score2, np.float32)
    tensorlist.append(attn_score1)
    tensorlist.append(attn_score2)
    tensorlist_stk = tf.stack(tensorlist)
    tensorlist.append(tf.zeros_like(tensorlist_stk[-1]))
    tensorlist_stk = tf.stack(tensorlist)

    print(attn_score1.get_shape())
    print(tensorlist_stk.get_shape())

    return

def tfstack_from_tensor():
    '''
    Experiement how to insert zero for a Tensor with shape (15, 256) to (16, 256)
    Returns:
    '''
    old_tensor = tf.random_normal(shape=[15, 256])
    zero_tensor = tf.zeros(shape=[1, old_tensor.get_shape().as_list()[1]])
    new_tensor = tf.concat([zero_tensor, old_tensor], axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nt = sess.run(new_tensor)
        print(nt[0,:])
    return

def tfstack_from_emptylist():
    tensorlist = []
    r = tf.stack(tensorlist)
    with tf.Session() as sess:
        r_ret = sess.run(r)
        print(r_ret)

def tfstack_from_none():
    t = None
    r = tf.stack(t)
    print(r)

def tfstack_from_none_in_list():
    tensorlist = [None]
    r = tf.stack(tensorlist)
    print(r)

if __name__=="__main__":
    tfstack_from_tensor()
    tfstack_from_tensorlist()
    tfstack_from_emptylist()
    #tfstack_from_none()
    '''
    Traceback (most recent call last):
  File "tf_stack_experiment.py", line 47, in <module>
    tfstack_from_none()
  File "tf_stack_experiment.py", line 36, in tfstack_from_none
    r = tf.stack(t)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 882, in stack
    value_shape = ops.convert_to_tensor(values[0], name=name).get_shape()
TypeError: 'NoneType' object is not subscriptable
    '''
    #tfstack_from_none_in_list()
    '''
    Traceback (most recent call last):
  File "tf_stack_experiment.py", line 58, in <module>
    tfstack_from_none_in_list()
  File "tf_stack_experiment.py", line 41, in tfstack_from_none_in_list
    r = tf.stack(tensorlist)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 882, in stack
    value_shape = ops.convert_to_tensor(values[0], name=name).get_shape()
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 836, in convert_to_tensor
    as_ref=False)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 926, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 229, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 208, in constant
    value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/tensor_util.py", line 371, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
Peters-MacBook-Pro:custom_project peli$
    '''

