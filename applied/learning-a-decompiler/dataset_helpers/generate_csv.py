import csv
import glob
import json

rows = []
for i in glob.glob("dataset/*.json"):
    with open(i, "r") as file:
        rows.append(json.load(file))


with open('dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['bin', 'source']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in rows:
        writer.writerow(i)
