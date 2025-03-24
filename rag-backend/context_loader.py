import os
from PyPDF2 import PdfReader

CHUNK_SIZE = 500

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([
            page.extract_text() for page in reader.pages if page.extract_text()
        ])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def chunk_text(text, size=CHUNK_SIZE):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < size:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks
