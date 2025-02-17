import glob
import json
from memory import Memory
from optimization_playground_shared.apis.openai import OpenAiCompletion
import os
import argparse

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

def get_concepts():
    concepts = {}
    path = os.path.join(
        os.path.dirname(__file__),
        "concepts/*.json"
    )
    for i in glob.glob(path):
        with open(i, "r") as file:
            data = json.load(file)
            data["path"] = i
            concepts[data["name"]] = data
    return concepts

concepts = get_concepts()

def load(args):    
    option = get_options(list(concepts.keys())) if args.concept is None else args.concept
    context = concepts[option]
    if not args.skip_memory:
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
    else:
        context_prompt = "\n".join(context["context"])
        model = OpenAiCompletion().chat_model(Memory.no_memory(context_prompt, args.prompt))
        print(model)

if __name__ == "__main__":
    concepts_keys = ",".join(list(concepts.keys()))
    parser = argparse.ArgumentParser(description="Assistance")
    parser.add_argument("--concept", type=str, required=False, help=f"concept to load({concepts_keys})")
    parser.add_argument("--skip-memory", action="store_true", default=False, help="Use memory for the prompt. Useful if you are going to talk to the agent multiple times.")
    parser.add_argument("--prompt",   required=False, default=None, help="Send prompt directly")

    args = parser.parse_args()
    load(args)
