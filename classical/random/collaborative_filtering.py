"""

https://en.wikipedia.org/wiki/Collaborative_filtering

"""
import numpy as np


def similarity(x, y):
    sum_xy = ((x * y).sum())
    normalize_x = np.sqrt((np.sum(x**2)))
    normalize_y = np.sqrt((np.sum(y**2)))
    return (sum_xy)/(normalize_x * normalize_y)


def normalize_factor(u, u_, current_user):
    user_sum = 0
    for i in range(u.shape[0]):
        if i != current_user:
            user_sum += np.abs(similarity(u[i], u_))
    return 1/user_sum


def all_user_similarly(x, u, item, current_user):
    similarity_score = 0
    for user in range(x.shape[0]):
        if user != current_user:
            raw_user_similarity = similarity(u, x[user])
            user_average_rating = np.sum(x[user]) / np.count_nonzero(x[user])
            current_item = x[user][item]
            item_delta = current_item - user_average_rating

            similarity_score += raw_user_similarity * item_delta
    return similarity_score


def aggregation(x, item=0):
    r = np.zeros(x.shape)
    for user in range(x.shape[0]):
        for item in range(x.shape[1]):
            if x[user, item] == 0:
                user_average_rating = np.sum(x[user]) / np.count_nonzero(x[user])
                k = normalize_factor(x, x[user], current_user=user)
                delta = user_average_rating + k * all_user_similarly(x, x[user], item, current_user=user)
                r[user, item] = delta
            else:
                r[user, item] = x[user, item]
    return r


if __name__ == "__main__":
    """
    1. Look for users who have similar patterns to the user to make predictions for
    2. Create prediction based on aggregations of the users
    """
    # Should not recommend the item
    x = np.array([
        [1, -1, 1, 1],
        [1, -1, 1, 1],
        [1, 0, 1, 1],
    ])
    x = aggregation(x)
    print(x)

    # Should recommend the item
    x = np.array([
        [1, 1, 1, -1],
        [1, 1, 1, 1],
        [1, 0, -1, 1],
    ])
    x = aggregation(x)
    print(x)
