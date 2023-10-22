from datasets import load_dataset

dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")

def preprocess_function(row):
    text = 256
    content = row["content"]
    inputs = content[:text]
    return inputs

def generate_some_text():
    for i in dataset["train"]:
        # I only want python
        if i["lang"] == "Python":
            yield preprocess_function(i)
            #print("")
