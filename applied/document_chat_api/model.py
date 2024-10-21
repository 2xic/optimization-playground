from optimization_playground_shared.apis.openai import OpenAiCompletion, OpenAiWhisper
from optimization_playground_shared.apis.google_ai_studio import get_gemini_output
import os
import requests
from string import Template 


# https://platform.openai.com/docs/models/gpt-4o-mini

speakers = {
    "female-1": "alloy",
    "male-1": "onyx",
    "female-2": "shimmer",
}

def get_text(id):
    url = os.environ["raw_host"] + f"/text/{id}"
    print(url)
    return requests.get(
        url, 
        cookies={
            "credentials": os.environ["auth_header"]
        }
    ).text

# https://platform.openai.com/docs/models
api = OpenAiCompletion(
#    "gpt-3.5-turbo",
    "gpt-4o-mini"
#    "gpt-4o-2024-08-06",
)
text = get_text(1750455)

prompt = None
assert os.path.isfile("prompt.txt")
with open("prompt.txt", "r") as file:
    prompt = Template(file.read()).substitute({
        "text": text,
    })
    
#print(prompt)
#exit(0)

#text_summarized = api.get_dialogue(
#    prompt,
#)
text_summarized = get_gemini_output(
    prompt
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

