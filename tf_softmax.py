import tensorflow as tf
def tf_softmax():
    # want to see softmax's softmax result
    logits = tf.random_normal(shape=[3,2])
    s1 = tf.nn.softmax(logits)
    s2 = tf.nn.softmax(s1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l, s1_np, s2_np = sess.run([logits, s1, s2])
        print("Before:")
        print(l)
        print("1st time:")
        print(s1_np)
        print("2nd time:")
        print(s2_np)

if __name__ == "__main__":
    tf_softmax()
    ''' conclusion, the values are changed after 2nd softmax
    Before:
[[ 0.09931096 -0.08985478]
 [ 0.9667916  -1.9911529 ]
 [ 0.5027439  -1.682604  ]]
1st time:
[[0.5471509  0.4528491 ]
 [0.95063764 0.04936238]
 [0.89892596 0.101074  ]]
2nd time:
[[0.523558   0.47644204]
 [0.7112115  0.2887885 ]
 [0.6895148  0.31048518]]
 '''
