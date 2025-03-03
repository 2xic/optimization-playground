import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GrokModel:
    def __init__(self, model="grok-2-latest"):
        self.model = model
        self.host = "https://api.x.ai/v1/chat/completions"
        self.api_key = os.environ["GROK_API_KEY"]

    def send_messages(self, messages):
        return requests.post(
            self.host,
            headers={
                f"Authorization": f"Bearer {self.api_key}"
            },
            json={
                "messages": messages,
                "model": self.model,
                "stream": False,
                "temperature": 0,
            }
        ).json()
    
if __name__ == "__main__":
    output = GrokModel()
    print(output.send_messages([
        {
        "role": "system",
        "content": "You are a test assistant."
        },
        {
        "role": "user",
        "content": "Testing. Just say hi and hello world and nothing else."
        }
    ]))
