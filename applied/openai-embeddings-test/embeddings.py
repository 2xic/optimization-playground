from dotenv import load_dotenv
import json
import glob
import time

load_dotenv()

from optimization_playground_shared.apis.openai_ada_embeddings import OpenAiAdaEmbeddings

model = OpenAiAdaEmbeddings()
# I want to study some OpenAi embeddings
# No clear goal, just visualize the embeddings in various contexts
# Actually, I want to study embeddings between spotify playlist.
# 

def get_spotify_lyrics():
    for file_path in glob.glob(".lyrics/*.json"):
        text = []
        data = None
        print(file_path)
        with open(file_path, "r") as file:
            data = json.load(file)
            if data["lyrics"]:
                for i in data["lyrics"]["lyrics"]["lines"]:
                    text.append(i["words"])
        if "embeddings" not in data:
            embeddings = model.get_embedding("\n".join(text))
            data["embeddings"] = embeddings
            with open(file_path, "w") as file:
                file.write(json.dumps(data))
            time.sleep(1)
            

if __name__ == "__main__":
    get_spotify_lyrics()

