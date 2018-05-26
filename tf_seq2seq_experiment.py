import tensorflow as tf
import numpy as np

scope_name = 'words'
def test_embed_sequence():
    '''
    embed_sequence is a initializer helper to covert your input sentence to embed_dim vectors
    Returns:

    '''
    vocab = [{'garbage':1}, {'piles':2}, {'in':3}, {'the':4}, {'city':5}, {'is':6}, {'clogged':7}, {'with':8}, {'vehicles':9}]
    features = [[1,2,3,4,5], [5,6,7,8,8]]

    EMBEDDING_SIZE = 10
    features_embedded = tf.contrib.layers.embed_sequence(
        ids = features,
        vocab_size = len(vocab),
        embed_dim=EMBEDDING_SIZE,
        scope=scope_name,
        reuse=tf.AUTO_REUSE
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        f_e = sess.run(features_embedded)
        print(f_e)
''' example output
[[[ 0.34158373 -0.54938513 -0.010243    0.22618032  0.08551908
    0.5068689   0.5036492   0.28661227  0.00457352  0.24749047]
  [-0.55297256 -0.49146557  0.38021672  0.29764014 -0.20319945
    0.24991119 -0.22958636  0.15095073 -0.474692    0.31670612]
  [ 0.12689322  0.54982775 -0.54190826  0.16049719  0.15744859
   -0.29497313  0.27991593  0.4888627   0.07495493  0.35160416]
  [-0.0428344   0.52511364  0.55380994  0.53304404 -0.3737836
   -0.00273991  0.25410318 -0.55734044 -0.26390445 -0.36016148]
  [ 0.544071    0.03993052  0.09873188 -0.46592072 -0.03088754
    0.10448605 -0.51889414  0.06264526  0.3221758  -0.02165943]]

 [[ 0.544071    0.03993052  0.09873188 -0.46592072 -0.03088754
    0.10448605 -0.51889414  0.06264526  0.3221758  -0.02165943]
  [ 0.03337097 -0.14310369  0.09040868  0.34511554  0.45794064
    0.33134246 -0.36586714 -0.27370775  0.18883765 -0.26836783]
  [ 0.18833345 -0.38724667  0.34714335 -0.52243847 -0.17981705
   -0.06192994 -0.47000483  0.114806    0.13825631  0.4995212 ]
  [ 0.5180362  -0.14453954  0.03119379  0.5563174   0.00471342
   -0.27514493 -0.11599892  0.16143757  0.55324274 -0.09431484]
  [ 0.5180362  -0.14453954  0.03119379  0.5563174   0.00471342
   -0.27514493 -0.11599892  0.16143757  0.55324274 -0.09431484]]]
'''

def test_read_embedding():
    print("Test read embedding from former defined embedding matrix.")

    with tf.variable_scope(scope_name, reuse=True):
        embedding_matrix = tf.get_variable(name='embeddings')
    dec_input = [[1,2,3],[4,5,6]]
    dec_embedded_input = tf.nn.embedding_lookup(embedding_matrix, dec_input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _embedded_input = sess.run(dec_embedded_input)
        print(_embedded_input)
''' sample output
Test read embedding from former defined embedding matrix.
[[[-0.47289813 -0.25815833 -0.31500334  0.24488401 -0.28925917
    0.13878876  0.2348941   0.41908973  0.14340913  0.18852264]
  [-0.5198182   0.52239734  0.50987273  0.5030301  -0.11493057
    0.39601523  0.29608607  0.18276781  0.5310734  -0.39876503]
  [-0.4901062   0.02497995  0.3326841   0.03895646  0.31780565
   -0.17664227 -0.34979734  0.03468615  0.17545146  0.12327039]]

 [[-0.3769814   0.07246828  0.29415274 -0.43250942 -0.32478
    0.06988913  0.35327882  0.1871854   0.01241922 -0.49669534]
  [-0.13383177  0.39311522 -0.29894897 -0.440247   -0.26188886
    0.48903102 -0.32403827 -0.36356026  0.238002    0.00801349]
  [ 0.0863629   0.07717121 -0.31319985  0.02371722  0.16456145
    0.25066954 -0.48035514  0.00727534  0.3620203  -0.3014355 ]]]
'''

if __name__ == "__main__":
    test_embed_sequence()
    test_read_embedding()