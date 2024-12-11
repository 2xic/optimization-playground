from optimization_playground_shared.apis.openai import OpenAiCompletion, OpenAiWhisper

def get_mp3(text):
    model = OpenAiWhisper()
    with  model.get_speech(text) as response:
        response.raise_for_status()
        with open("output.mp3", 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
