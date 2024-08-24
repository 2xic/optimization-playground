"""
https://vigneshwarar.substack.com/p/hackernews-ranking-algorithm-how
https://medium.com/hacking-and-gonzo/how-hacker-news-ranking-algorithm-works-1d9b0cf2c08d
"""
import matplotlib.pyplot as plt 

def rank(points, age_hours, G=1.8):
    return (
        points
    ) / (age_hours + 2) ** G


def generate_scores(incremental_up, incremental_down, times):
    scores = []
    up = incremental_up
    down = incremental_down
    X = []
    for index, age_hours in enumerate(times):
#        scores.append(log(hot(up, down, age_hours)))
        scores.append(rank((up - down), age_hours))
        up += incremental_up
        down += incremental_down
        X.append(-(len(times) - index))
    return X, scores

if __name__ == "__main__":
    times = []
    for age_hours in range(48, 0, -1):
        times.append(age_hours)
    for up, down in [
        (10, 0),
        (100, 200),
        (100, 50)
    ]:
        plt.plot(*generate_scores(up, down, times), label=f"Up {up}, down {down} each hour")
    plt.legend(loc="upper left")
    plt.ylabel("Score")
    plt.xlabel("Hours (1 day ago to one hour ago)")
    plt.savefig("hn_ranking_over_time.png")
