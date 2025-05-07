from optimization_playground_shared.apis.openai import OpenAiCompletion
from optimization_playground_shared.apis.openai_helper import get_web_searches
import argparse
import fitz
import base64
import requests
import os
import json

if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Give an url and get the summary")
 #   parser.add_argument("url", type=str, help="The url argument")
  #  args = parser.parse_args()

    text = get_web_searches("What is new on twitter today in Machine learning? I just want papers or good blogs.")
    print(json.dumps(text.json()))

 #   text = get_web_searches("What is new on twitter today in crypto ? I just want papers or good blogs.")
#    print(text.json())
