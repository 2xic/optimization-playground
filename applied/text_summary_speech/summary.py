from optimization_playground_shared.apis.openai import OpenAiCompletion
import os
import requests
import sys
import argparse

def get_id(url):
    url = os.environ["raw_host"] + f"url/id?url={url}"
    print(url)
    return requests.get(
        url, 
        cookies={
            "credentials": os.environ["auth_header"]
        }
    ).json()["id"]

def get_text(id):
    if id is None:
        return None
    url = os.environ["raw_host"] + f"/text/{id}"
    print(url)
    return requests.get(
        url, 
        cookies={
            "credentials": os.environ["auth_header"]
        }
    ).text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give an url and get the summary")
    parser.add_argument("url", type=str, help="The url argument")
    args = parser.parse_args()

    api = OpenAiCompletion()
    url_id = get_id(args.url)
    text = get_text(url_id)

    if text is None:
        print(f"I found no text on that page.")
    else:
        text_summarized = api.get_summary(
            text,
        )
        print(text_summarized)
