from dotenv import load_dotenv
import requests
import os
load_dotenv()

def get_tweets():
    tweets = requests.get(
        os.environ.get("HOST")
    ).json()
    data ,users = list(map(lambda x: x["tweet"]["text"], tweets)),\
            list(map(lambda x: x["tweet"].get("user", {"name":"???"})["name"], tweets))
    top_n_users = 5
    group_data_users = {}
    for (tweet, user) in zip(data, users):
        if not user in group_data_users:
            group_data_users[user] = []
        group_data_users[user].append(tweet)
    data = []
    users = []
    for key, value in sorted(group_data_users.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_users]:
        data += value
        for _ in range(len(value)):
            users.append(key)
    return data, users

print(get_tweets())
