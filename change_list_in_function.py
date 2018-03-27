def func_change_list(i, l):
    '''
    test change list in this func
    Args:
        l: list to change

    Returns:

    '''
    print("l[-1] {}".format(l[-1]))
    l.append(i)

def test():
    ll = [1]
    for k in range(5,10,1):
        func_change_list(k, ll)

if __name__ == "__main__":
    test()
