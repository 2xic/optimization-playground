from utils import LossTracker
import json

data = LossTracker('test_a')
#data.load()

with open("encoder/loss.json", "r") as file:
    data.loss = json.load(file)["Loss"]

data.save(
    y_scale='symlog'
)

