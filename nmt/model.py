'''
The NMT model file
'''
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *

'''
1 - Translating human readable dates into machine readable dates

The network will input a date written in a variety of possible formats (e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987") and translate them into standardized, machine readable dates (e.g. "1958-08-29", "1968-03-30", "1987-06-24"). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.
'''
def test_load_data():
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

if __name__ == "__main__":
    test_load_data()