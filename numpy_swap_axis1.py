import numpy as np

def test_numpy_array_swap():
    '''
    Give a np.array np_arr with shape [batch_size, 2], want to swap the second dimension values
    np_arr[:,0] --> np_arr[:,1]
    np_arr[:,1] --> np_arr[:,0]

    This can done by advanced slicing, see code.

    Returns: None
    '''
    np_arr = np.random.randn(5,2)
    print("Before swap---\n{}".format(np_arr))
    np_arr[:,[0,1]] = np_arr[:,[1,0]]
    print("After swap----\n{}".format(np_arr))

if __name__ == "__main__":
    test_numpy_array_swap()