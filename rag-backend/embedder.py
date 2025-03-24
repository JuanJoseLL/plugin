import requests

EMBEDDING_SERVICE_URL = "http://216.81.248.42:8001/embed"

def get_embedding(text: str):
    response = requests.post(EMBEDDING_SERVICE_URL, json={"text": text})
    return response.json()["embedding"]
