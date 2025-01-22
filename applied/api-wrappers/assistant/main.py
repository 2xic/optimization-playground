import glob
import json
from memory import Memory
from optimization_playground_shared.apis.openai import OpenAiCompletion
import os

def get_options(options):
    print("Concepts:")
    for index, i in enumerate(options):
        print(f"{index}. {i}")
    print("")
    id = input("Which option do you want to load? (eg. 1 or 2)")
    if id.isnumeric():
        id = int(id)
    else:
        id = -1
    print(id)
    if id < 0 or id > len(options):
        print("Invalid option")
        return get_options(options)
    return options[id]

def load():
    concepts = {}
    for i in glob.glob("concepts/*.json"):
        with open(i, "r") as file:
            data = json.load(file)
            data["path"] = i
            concepts[data["name"]] = data
    
    option = get_options(list(concepts.keys()))
    context = concepts[option]
    memory = Memory(os.path.basename(context["path"]))
    memory = memory.create_summary()
    context_prompt = "\n".join(context["context"])
    while True:
        model = OpenAiCompletion().chat_model(memory.get_context(context_prompt))
        print(model)
        memory.add_system(model)
        response = input("> ")
        memory.add_user(response)
        memory.save()

if __name__ == "__main__":
    load()
