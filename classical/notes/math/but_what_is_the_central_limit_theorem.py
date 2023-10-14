"""
https://www.youtube.com/watch?v=zeJD6dqJ5lo
"""
import matplotlib.pyplot as plt 
import random
n_dices = 10
times = 10_000

results = {}
for i in range(times):
    sum = 0
    for i in range(n_dices):
        sum += random.randint(1, 6)
    results[sum] = results.get(sum, 0) + 1
 
sum = list(results.keys())
count = list(results.values())
  
fig = plt.figure(figsize = (10, 5))
 
plt.bar(sum, count, width = 0.4)
 
plt.xlabel("Sum")
plt.ylabel("No. of times we got the value")
plt.title(f"Sum of {n_dices} thrown {times}")
plt.savefig('central_limit.png')
