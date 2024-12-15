"""
I accidentally deleted the source db during a migration, but can retrieve most of the metadata from my the active cache.

https://arrow.apache.org/docs/python/parquet.html
"""
import glob
import json

rows = []


for i in glob.glob(".cache/*"):
    with open(i, "r") as file:
        data = json.load(file)
        for i in data.get("entries", []):
            #print(i)
            entry = {
                "text": i["text"],
                "is_good_content": i["is_good"],
            }
            if entry not in rows and len(entry["text"]) > 0:
                rows.append(entry)

with open("training_data.json", "w") as file:
    json.dump(rows, file, indent=4)
