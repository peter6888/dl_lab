'''
The implementations when reading this article
https://www.topcoder.com/community/data-science/data-science-tutorials/dynamic-programming-from-novice-to-advanced/
'''
import sys
def q1_coins(sumto = 11, coins = [1,3,5]):
    states = [sys.maxsize for _ in range(sumto + 1)]
    results = [list() for _ in range(sumto + 1)]
    states[0] = 0
    for i in range(1, sumto+1): # update sumto steps
        for coin in coins:
            if coin <= i and states[i-coin] + 1 < states[i]:
                states[i] = states[i-coin] + 1
                results[i] = results[i-coin] + [coin]
    return (states[sumto], results[sumto])

def q2_zigzag(a):
    '''
    find max length of zigzag list
    Args:
        a: list(int)
    Returns: the max length
    '''
    n = len(a)
    if n in [0,1,2]:
        return n

    ans = [2 for _ in range(n)]
    ans[0], ans[1] = 1, 2

    for i in range(2, n):
        for j in range(1, i):
            if (a[j] - a[j-1]) * (a[i] - a[j]) < 0 and \
                                    ans[j] + 1 > ans[i]:
                ans[i] = ans[j] + 1
    return ans[-1]

def q3_badneighbors(donations):
    '''
    Find max donations.
    Solution idea: First solve this problem without circle, it's a normal dynamic-problem.
        Then, consider the neighors at begining and end, it can fixed by zero out the smaller one.
    Args:
        donations: list(int)

    Returns:max donation
    '''
    n = len(donations)
    if n <= 2:
        return max(donations)

    # zero out one smaller neighbors at begining or end to avoid circle issue
    if donations[0] > donations[-1]:
        donations[-1] = 0
    else:
        donations[0] = 0

    ans = list(donations)
    ans[1] = max(donations[:2])

    for i in range(2, n):
        ans[i] = max(ans[i-1], donations[i]+ans[i-2])

    return ans[-1]

if __name__ == "__main__":
    print("q1 coins default input:{}".format(q1_coins()))
    print("q1 coints 20 with coins [1,5,7], output:{}".format(q1_coins(20, [1,5,7])))
    print("q2 [1,2,1], output {}".format(q2_zigzag([1,2,1])))
    q2_ts = [[1, 7, 4, 9, 2, 5], \
             [1, 17, 5, 10, 13, 15, 10, 5, 16, 8], \
             [44], \
             [1, 2, 3, 4, 5, 6, 7, 8, 9], \
             [70, 55, 13, 2, 99, 2, 80, 80, 80, 80, 100, 19, 7, 5, 5, 5, 1000, 32, 32], \
             [374, 40, 854, 203, 203, 156, 362, 279, 812, 955,
600, 947, 978, 46, 100, 953, 670, 862, 568, 188,
67, 669, 810, 704, 52, 861, 49, 640, 370, 908,
477, 245, 413, 109, 659, 401, 483, 308, 609, 120,
249, 22, 176, 279, 23, 22, 617, 462, 459, 244]]
    for q2_t in q2_ts:
        print("q2 {} output {}".format(q2_t, q2_zigzag(q2_t)))

    q3_ts = [[ 10, 3, 2, 5, 7, 8], \
             [11, 15], \
             [15, 11], \
             [15, 11, 12], \
             [ 7, 7, 7, 7, 7, 7, 7 ], \
              [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], \
             [94, 40, 49, 65, 21, 21, 106, 80, 92, 81, 679, 4, 61,
  6, 237, 12, 72, 74, 29, 95, 265, 35, 47, 1, 61, 397,
  52, 72, 37, 51, 1, 81, 45, 435, 7, 36, 57, 86, 81, 72]
        ]

    for q3_t in q3_ts:
        print("q3 {} output {}".format(q3_t, q3_badneighbors(q3_t)))

