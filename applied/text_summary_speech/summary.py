from optimization_playground_shared.apis.openai import OpenAiCompletion, OpenAiWhisper
import os
import requests

def get_text(id):
    url = os.environ["raw_host"] + f"/text/{id}"
    print(url)
    return requests.get(
        url, 
        cookies={
            "credentials": os.environ["auth_header"]
        }
    ).text

api = OpenAiCompletion()
text = get_text(1750455)
print(len(text))

text_summarized = api.get_summary(
    text,
)
print(text_summarized)
exit(0)

model = OpenAiWhisper()

with  model.get_speech(text_summarized) as response:
    response.raise_for_status()
    with open("output.mp3", 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
