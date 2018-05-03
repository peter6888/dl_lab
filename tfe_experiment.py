'''
Experiement to use tensorflow Eager execution. 
'''
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

x = [[2.0]]
m = tf.matmul(x,x)
print(m)