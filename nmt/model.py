'''
The NMT model file
'''
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import load_dataset, preprocess_data
import tensorflow as tf
import numpy as np

'''
1 - Translating human readable dates into machine readable dates

The network will input a date written in a variety of possible formats (e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987") and translate them into standardized, machine readable dates (e.g. "1958-08-29", "1968-03-30", "1987-06-24"). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.
'''
def test_load_data():
    load_data()

def load_data():
    m = 10000
    print("get dataset")
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
    print("dataset[:10]", dataset[:10])
    '''
    You've loaded:
        dataset: a list of tuples of (human readable date, machine readable date)
        human_vocab: a python dictionary mapping all characters used in the human readable dates to an integer-valued index
        machine_vocab: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with human_vocab.
        inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to characters.
    Let's preprocess the data and map the raw text data into the index values. We will also use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we would have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long).
    '''
    Tx = 30
    Ty = 10
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Xoh.shape:", Xoh.shape)
    print("Yoh.shape:", Yoh.shape)

    return X, Y, Xoh, Yoh

''' part of sample output
.....
.....
X.shape: (10000, 30)
Y.shape: (10000, 10)
Xoh.shape: (10000, 30, 37)
Yoh.shape: (10000, 10, 11)
'''

class model(object):
    def __init__(self):
        self.batch_size = 10000
        self.enc_seq_length = 30
        self.enc_emb_size = 37
        self.dec_seq_length = 10
        self.dec_emb_size = 11

    def _placeholder(self):
        self.enc_inputs = tf.placeholder(shape=[self.batch_size, self.enc_seq_length, self.enc_emb_size], dtype=tf.int32)
        self.dec_inputs = tf.placeholder(shape=[self.batch_size, self.dec_seq_length, self.dec_emb_size], dtype=tf.int32)

    def build_feeddict(self, encoder_inputs, decoder_inputs=None):
        self.feeddict = {self.enc_inputs: encoder_inputs}
        if decoder_inputs is not None:
            self.feeddict[self.dec_inputs] = decoder_inputs

    def _encoder_layer(self):
        pass

    def _decoder_layer(self):
        pass

    def optimizer_op(self):
        pass

    '''
    1. Placeholder
    2. Encoder-Decoder with Attention
    3. Loss
    Returns:

    '''
class basic_seq2seq2_model(model):
    pass

'''
Task 1. To build a basic seq2seq model use Tensorflow seq2seq2 API(s)
Task 2. Train the model done in Task 1, note the loss and converge status
Task 3. Use Attention API in Tensorflow seq2seq API(s)
Task 4. Train the Task 3 model, which should have better result than Task 2 (s)
'''

def test_basic_model():
    basic_model = model()
    basic_model._placeholder()
    X, Y, Xoh, Yoh = load_data()
    basic_model.build_feeddict(Xoh, Yoh)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _enc, _dec = sess.run([basic_model.enc_inputs, basic_model.dec_inputs], feed_dict=basic_model.feeddict)
        print("encoder inputs shape {}, decoder inputs shape {}".format(_enc.shape, _dec.shape))

if __name__ == "__main__":
    #test_load_data()
    test_basic_model()