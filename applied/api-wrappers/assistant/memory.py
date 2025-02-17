import os 
import json
from optimization_playground_shared.apis.openai import OpenAiCompletion

cache_history = os.path.join(
    os.path.dirname(__file__),
    "history"
)
os.makedirs(cache_history, exist_ok=True)

class Memory:
    def __init__(self, name):
        self.path = os.path.join(
            cache_history,
            name
        )
        self.history = []
        if os.path.isfile(self.path):
            with open(self.path, "r") as file:
                self.history = json.load(file)

    def create_summary(self):
        if len(self.history) > 0:
            summary_prompt = OpenAiCompletion().get_summary(
                "\n".join(self.flatten_history())
            )
            self.history = [
                self.history[0],
                {
                    "role": "system",
                    "content": "summary so far: " +summary_prompt,
                }
            ]
        return self 
    
    @classmethod
    def no_memory(self, context, prompt):
        return [
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

    def get_context(self, prompt):
        if len(self.history) == 0:
            return [
                {
                    "role": "system",
                    "content": prompt,
                },
            ]
        else:
            return self.history

    def flatten_history(self):
        messages = []
        for i in self.history:
            messages.append(i["content"])
        return messages

    def add_system(self, message):
        self.history.append({
            "role": "system",
            "content": message,
        })

    def add_user(self, message):
        self.history.append({
            "role": "user",
            "content": message,
        })

    def save(self):
        with open(self.path, "w") as file:
            json.dump(self.history, file)
