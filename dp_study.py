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

if __name__ == "__main__":
    print("q1 coins default input:{}".format(q1_coins()))
    print("q1 coints 20 with coins [1,5,7], output:{}".format(q1_coins(20, [1,5,7])))

