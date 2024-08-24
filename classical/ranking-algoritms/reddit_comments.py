"""
https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9#.uc7audaur
"""
import numpy as np
import matplotlib.pyplot as plt 

def rank(points, age_hours, G=1.8):
    return (
        points
    ) / (age_hours + 2) ** G


def generate_scores(ups, downs):
    n = ups + downs
    if n == 0:
        return 0

    z = 1.281551565545
    p = float(ups) / n

    left = p + 1/(2*n)*z*z
    right = z * np.sqrt(p * (1-p)/n + z*z/(4*n*n))
    under = 1+1/n*z*z

    return (left - right) / under

if __name__ == "__main__":
    results = {}
    for up, down in [
        [0, 10],
        [10, 10],
        [50, 10],
        [100, 0],
        [50, 25],
        [10, 5]
    ]:
        results[f"Up {up}, down {down}"] = generate_scores(up, down)
        
    fig = plt.figure(figsize = (10, 5))
    plt.bar(results.keys(), results.values(), width = 0.4)
    
    plt.xlabel("Sum")
    plt.ylabel("No. of times we got the value")    
    plt.savefig("reddit_comments.png")
