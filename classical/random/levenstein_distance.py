import numpy as np
# from https://stackoverflow.com/a/32558749
def levenshtein_distance(s1, s2):
    assert type(s1) == str
    assert type(s2) == str
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    print(s1, s2)
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
