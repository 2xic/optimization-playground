from optimization_playground_shared.apis.openai import OpenAiCompletion
from optimization_playground_shared.apis.url_to_text import get_text
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give an url and get the summary")
    parser.add_argument("url", type=str, help="The url argument")
    args = parser.parse_args()

    api = OpenAiCompletion()
    text = get_text(args.url)

    if text is None:
        print(f"I found no text on that page.")
    else:
        text_summarized = api.get_summary(
            text,
        )
        print(text_summarized)
