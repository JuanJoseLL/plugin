from fastapi import FastAPI, UploadFile, Form
from context_loader import extract_text, chunk_text
from embedder import get_embedding
from indexer import create_faiss_index
import os
from fastapi.middleware.cors import CORSMiddleware
from model_client import ask_qwen
from retriever import search_similar_chunks

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-context")
async def upload_context(file: UploadFile):
    file_path = f"data/context_files/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(file_path)
    chunks = chunk_text(text)

    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    create_faiss_index(embeddings, chunks)

    return {"status": "ok", "chunks": len(chunks)}

@app.post("/chat")
async def chat(message: str = Form(...)):

    query_embedding = get_embedding(message)

    top_chunks = search_similar_chunks(query_embedding)

    context = "\n\n".join(top_chunks)
    system_prompt = "Eres un asistente útil que responde únicamente con base en el contexto proporcionado. Si no encuentras información relevante, indica que no puedes responder."

    reply = ask_qwen(system_prompt, context, message)

    return { "reply": reply }