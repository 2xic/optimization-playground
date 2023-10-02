import math
"""
Probability of word usage
"""
from collections import defaultdict
def probability_of_words(sentence):
    prob = defaultdict(float)
    count_letter = defaultdict(int)
    for i in sentence:
        prob[i] += 1
        count_letter[i] += 1
    for i in prob:
        prob[i] /= len(sentence) 
    print("Chars sorted by highest probability first")
    for key, value in sorted(prob.items(), key=lambda x: x[1], reverse=True):
        print(f"{key} : {value}")
    return prob, count_letter
# An ensemble (char and outcome) 
prob, count_letter = probability_of_words("hhhhheeeeellloooo! I like bagels")
# abbreviations
print(f"p(h) = {prob['h']}")
# probability of a subset
print(f"p(vowels) = {prob['a'] + prob['e'] + prob['i'] + prob['o']}")
# join ensemble 
print(f"p(a, b) = {prob['a'] + prob['b']}")
# marginal probability

"""
Conditional probability
"""
from collections import defaultdict
def conditional_probability(sentence):
    bi_gram = defaultdict(float)
    for index, i in enumerate(sentence[:-1]):
        bigram = f"{i}" + sentence[index + 1]
        bi_gram[bigram] += 1
    sum_p = sum(bi_gram.values())
    #for i in bi_gram:
    #    bi_gram[i] /= sum_p
    print("Chars sorted by highest probability first")
    for key, value in sorted(bi_gram.items(), key=lambda x: x[1], reverse=True):
        print(f"{key} : {value}")
    return bi_gram
bigram_p = conditional_probability("hhhhheeeeellloooo! I like bagels")
print(f"P(h | h) = {bigram_p['hh'] / count_letter['h']}")
print(f"P(b | a) = {bigram_p['ba'] / count_letter['a']}")
# product rule
print(f"P(b and a) = {(bigram_p['ba'] / count_letter['a']) * prob['a']}")
print(f"P(h and h) = {(bigram_p['hh'] / count_letter['h']) * prob['h']}")
# independence
# P(x, y) = P(X) * P(Y)
print(f"P(b, a) = {prob['b'] * prob['a']}")

"""
Entropy
"""
from collections import defaultdict
def entropy_of_words(sentence):
    prob = defaultdict(float)
    entropy = defaultdict(int)
    for i in sentence:
        prob[i] += 1
    for i in prob:
        prob[i] /= len(sentence) 
        entropy[i] = math.log2(1 / prob[i])
    print("Chars sorted by highest probability first")
    for key, value in sorted(entropy.items(), key=lambda x: x[1], reverse=True):
        print(f"{key} : {value}")
    return prob, entropy
prob, entropy = entropy_of_words("hhhhheeeeellloooo! I like bagels")
# entropy of an ensemble
print(sum([
    prob[i] * entropy[i]
    for i in entropy
]))
prob, entropy = entropy_of_words("hhhhhhhhhhhhhhhhhhhhhhhhh")
# entropy of an ensemble
print(sum([
    prob[i] * entropy[i]
    for i in entropy
]))
prob, entropy = entropy_of_words("hahahahahahahahaha1")
# entropy of an ensemble
print(sum([
    prob[i] * entropy[i]
    for i in entropy
]))