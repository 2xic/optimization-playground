from .openai_embeddings import get_embeddings
import numpy as np
import tiktoken

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

def get_text_batches(text):
    last_index = 0
    batches = []

    while last_index < len(text):
        batch_text, last_index = scale_down(text, last_index, max_token=8191)
        batches.append(batch_text)
    return batches
        

class OpenAiAdaEmbeddings:
    def __init__(self):
        self.model = "text-embedding-ada-002"

    def get_embedding(self, X: str):
        batches = get_text_batches(X)
        response = None
        for i in batches:
            if response is None:
                response = self._get_embeddings(
                    text=i,
                )
            else:
                response += self._get_embeddings(
                    text=i,
                )
        response /= len(batches)
        return response.tolist()

    def _get_embeddings(self, text):
        response = get_embeddings(
                text=text,
                model=self.model,
        )
        error = response.get("error", None)
        if not error is None:
            print(error)
            return error
        return np.asarray(response["data"][0]["embedding"]).astype(np.float32)
