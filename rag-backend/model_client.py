import requests
import os
from dotenv import load_dotenv

load_dotenv()

QWEN_API_URL = "https://api.totalgpt.ai/v1/chat/completions"
API_KEY = os.getenv("INFERMATIC_API_KEY")

MODEL_NAME = "Sao10K-72B-Qwen2.5-Kunou-v1-FP8-Dynamic"

def ask_qwen(system_prompt, context, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\n{question}"}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 7000,
        "temperature": 0.7,
        "top_k": 40,
        "repetition_penalty": 1.2
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]
