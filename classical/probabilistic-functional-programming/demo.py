"""
Same example from https://web.engr.oregonstate.edu/~erwig/pfp/
"""
from distribution import UniformDistribution, BinomialDistribution
die = UniformDistribution(
    list(range(1, 6 + 1))
)
print(die)

"""
Either adds 1 or not the number given
"""
print("=" * 12)
succOrId = UniformDistribution([
    lambda x: x,
    lambda x: x + 1
])
print(succOrId)

"""
What is the distribution of points when rolling two dice? 
"""
print("=" * 12)
die = UniformDistribution(
    list(range(1, 6 + 1))
)
results = die.prod(die)
print(results)

"""
What is the distribution of points when rolling two dice? 
"""
print("=" * 12)
die_1 = UniformDistribution(
    list(range(1, 5 + 1))
)
die_2 = UniformDistribution(
    list(range(1, 4 + 1))
)
results = die_1.join(die_2)
print(results)

""""
Binomial
"""

coin_flip = BinomialDistribution([
    0.3, 0.7
])
experiment = coin_flip.outcomes_after_n(6)
for i in experiment:
    print(i)
    print("")