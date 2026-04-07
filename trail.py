import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def query_gemma(prompt: str, stream: bool = False) -> str:
    payload = {
        "model": "gemma4:latest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

print(query_gemma("What is the capital of France?"))
