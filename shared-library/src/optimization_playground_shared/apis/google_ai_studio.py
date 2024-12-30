import requests
import os 

YOUR_API_KEY = os.environ["GOOGLE_AI_STUDIO"]

def get_gemini_output(text):
    return requests.post(
#        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={YOUR_API_KEY}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={YOUR_API_KEY}",
        json={
            "contents":[
                {
                    "parts":[
                        {
                            "text": text,
                        }
                    ]
                }
            ],
            "generationConfig": {
#                "stopSequences": [
#                    "Title"
#                ],
#                "temperature": 1.0,
#                "maxOutputTokens": 100_800,
#                "topP": 0.8,
#                "topK": 10
            }
        }
    ).json()
