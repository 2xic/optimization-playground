from optimization_playground_shared.apis.openai import OpenAiCompletion, get_mp3_from_text
from optimization_playground_shared.apis.google_ai_studio import get_gemini_output
from optimization_playground_shared.apis.url_to_text import get_text
import os
from string import Template 
import argparse

# https://platform.openai.com/docs/models/gpt-4o-mini

speakers = {
    "female-1": "alloy",
    "male-1": "onyx",
    "female-2": "shimmer",
}

# https://platform.openai.com/docs/models
api = OpenAiCompletion("gpt-4o-mini")

def get_model_input(url):
    text = get_text(url)
    assert os.path.isfile("prompt.txt")
    with open("prompt.txt", "r") as file:
        return Template(file.read()).substitute({
            "text": text,
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give an url and get the summary")
    parser.add_argument("url", type=str, help="The url argument")
    args = parser.parse_args()
    text_input = get_model_input(args.url)
    text_summarized = get_gemini_output(text_input)
    text_content = text_summarized["candidates"][0]["content"]["parts"][0]["text"]
    output = get_mp3_from_text(text_content)

    with open("output.mp3", "wb") as file:
        file.write(output.getvalue())
