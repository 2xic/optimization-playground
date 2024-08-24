"""
https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9#.uc7audaur
https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
https://github.com/reddit-archive/reddit/blob/753b17407e9a9dca09558526805922de24133d53/r2/r2/lib/db/_sorts.pyx#L57
"""
from datetime import datetime, timedelta
from math import log
import matplotlib.pyplot as plt 

epoch = datetime(1970, 1, 1)

def epoch_seconds(date):
    td = date - epoch
    seconds_in_hours = 3600
    days_in_sends = 24 * seconds_in_hours
    return td.days * days_in_sends + td.seconds + (float(td.microseconds) / 1000000)

def score(ups, downs):
    return ups - downs

def sign(value):
    if value == 0:
        return 0 
    elif value < 0:
        return -1
    else:
        return 1

def hot(ups, downs, date):
    # https://www.quora.com/Where-do-the-constants-1134028003-and-45000-come-from-in-reddits-hotness-algorithm
    oldest_submission_unix_timestamp = 1134028003
    half_a_day_seconds = 45000 # 12.5 hours to be precise
    s = score(ups, downs)
    order = log(max(abs(s), 1), 10)
    # no idea about these constants
    seconds = epoch_seconds(date) - oldest_submission_unix_timestamp
    # no idea about these constants
    return round(sign(s) * order + seconds / half_a_day_seconds, 7)

def generate_scores(incremental_up, incremental_down, times):
    scores = []
    up = incremental_up
    down = incremental_down
    X = []
    for index, v in enumerate(times):
#        scores.append(log(hot(up, down, v)))
        scores.append((hot(up, down, v)))
        up += incremental_up
        down += incremental_down
        X.append(-(len(times) - index))
    return X, scores

if __name__ == "__main__":
    times = []
    for v in range(24, 0, -1):
        times.append(datetime.now() - timedelta(hours=v))
    for up, down in [
        (10, 0),
        (100, 200),
        (100, 50)
    ]:
        plt.plot(*generate_scores(up, down, times), label=f"Up {up}, down {down} each hour")
    plt.legend(loc="upper left")
    plt.ylabel("Score")
    plt.xlabel("Hours (1 day ago to one hour ago)")
    plt.savefig("reddit_ranking_over_time.png")
