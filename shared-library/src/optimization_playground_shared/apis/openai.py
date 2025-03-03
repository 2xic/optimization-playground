from .openai_helper import get_embeddings, get_completion, get_text_to_speech
import numpy as np
import tiktoken
from io import BytesIO
from pydub import AudioSegment
import os 

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#num_tokens_from_string("tiktoken is great!", "cl100k_base")

def scale_down(text, last_index, max_token):
    scale = lambda x:  text[last_index:x]
    current_index = 0
    current = scale(current_index)
    while (last_index + current_index) < len(text):
        next_entry = scale(current_index)
        if num_tokens_from_string(next_entry, "cl100k_base") < max_token:
            current_index += 64
            current = next_entry
        else:
            break
    return current, last_index + current_index

def get_text_batches(text, max_tokens=8191):
    last_index = 0
    batches = []

    while last_index < len(text):
        batch_text, last_index = scale_down(text, last_index, max_token=max_tokens)
        batches.append(batch_text)
    return batches
        

class OpenAiEmbeddings:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model

    def get_embedding(self, X: str):
        batches = get_text_batches(X)
        response = None
        for i in batches:
            if response is None:
                response = self._get_text(
                    text=i,
                )
            else:
                response += self._get_text(
                    text=i,
                )
        response /= len(batches)
        return response.tolist()

    def _get_text(self, text):
        response = get_embeddings(
                text=text,
                model=self.model,
        )
        error = response.get("error", None)
        if not error is None:
            print(error)
            return error
        return np.asarray(response["data"][0]["embedding"]).astype(np.float32)

    def name(self):
        return self.model

class OpenAiCompletion:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.tokens_length = {
            "gpt-3.5-turbo": 16385,
            "gpt-4o-mini": 128 * 1_000,
            "gpt-4o": 128 * 1_000,
            "gpt-4o-2024-08-06": 128 * 1_000,
        }

    def get_summary(self, text: str):
        # First summarize all the batches
        text = self._get_summary(text)
        # Then fix all the combined one to one summarization
        text = self._get_summary(text)
        return text

    def get_dialogue(self, text: str):
        batches = get_text_batches(text, max_tokens=self.tokens_length[self.model])
        responses = []
        for _, chunk in enumerate(batches):
            messages = [
                {
                    "role": "system",
                    "content": chunk
                }
            ]
            response = self._get_completions(
                messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":"items",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "properties": {
                                            "speaker": { "type": "string" },
                                            "text": { "type": "string" }
                                        },
                                        "required": ["speaker", "text"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                        }
                    }
                } if False and "gpt-4o-mini" in self.model else None
            )
            responses.append(response)
        return responses

    def _get_summary(self, text: str):
        batches = get_text_batches(text)
        responses = []
        length = len(batches)
        for index, chunk in enumerate(batches):
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that summarizes text. This is chuck {index} / {length}. Please make it as informative as possible."
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
            response = self._get_completions(
                messages,
                response_format=None,
            )
            responses.append(response)
        return "\n".join(responses)

    def chat_model(self, messages):
        response = self._get_completions(
            messages,
            response_format=None,
        )
        return response

    def _get_completions(self, messages, response_format):
        response = get_completion(
            messages=messages,
            model=self.model,
            response_format=response_format,
        )
        error = response.get("error", None)
        if not error is None:
            print("error ", error)
            return error
        # print(response)
        return response["choices"][0]["message"]["content"]

    def name(self):
        return self.model

class OpenAiWhisper:
    def __init__(self) -> None:
        pass

    def get_speech(self, text):
        return get_text_to_speech(text)


import re

def _is_all_whitespace(s):
    return bool(re.match(r'^\s*$', s))

def get_mp3_from_text(text):
    assert type(text) == str, type(text)
    model = OpenAiWhisper()
    combined = AudioSegment.empty()
    for text_batch in get_text_batches(text, max_tokens=512):
        if _is_all_whitespace(text_batch):
            continue
        data = bytes()
        print(text_batch)
        with model.get_speech(text_batch) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    # bytes_io.write(chunk)
                    data += chunk
        audio = AudioSegment.from_file(BytesIO(data), format="mp3")
        combined += audio

    bytes_io = BytesIO()
    combined.export(bytes_io, format="mp3")
    return bytes_io
