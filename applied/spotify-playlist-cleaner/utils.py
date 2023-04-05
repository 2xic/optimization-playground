import random

def shuffle(x, y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1, res2
