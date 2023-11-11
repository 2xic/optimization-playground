from datasets import load_dataset

dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")

def preprocess_function(row):
    content = row["content"]
    inputs = content.split(" ")[:512]
    return " ".join(inputs)

def generate_some_text():
    for i in dataset["train"]:
        # I only want python
        if i["lang"] == "Python":
            yield preprocess_function(i)            
