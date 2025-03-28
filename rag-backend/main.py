from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# Import components and config (adjust paths if needed)
from config import (logger, UPLOAD_DIR, TOP_K_INITIAL_SEARCH)
from embedder import AzureEmbeddings
from model_client import CustomChatQwen
from graph_db import get_neo4j_graph_instance, get_neo4j_vector_store
from document_processing import load_and_split_document
from retriever import get_graph_enhanced_retriever, format_docs
import os
# --- Initialize global components (or use FastAPI dependencies) ---
# These could be initialized once and passed via Depends for better management
try:
    embedding_model = AzureEmbeddings()
    chat_model = CustomChatQwen()
    neo4j_graph = get_neo4j_graph_instance() # Ensures constraints/indices
    neo4j_vector_store = get_neo4j_vector_store(embedding_model)
    logger.info("Initialized LangChain components (Embeddings, ChatModel, Neo4jGraph, Neo4jVector)")
except Exception as e:
    logger.exception(f"Fatal error during initialization: {e}")
    # Application might not be usable, handle appropriately (e.g., exit or raise)
    raise RuntimeError(f"Failed to initialize core components: {e}")


# --- FastAPI App Setup ---
app = FastAPI(title="LangChain GraphRAG Agent", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Background Task for Ingestion ---
def ingest_pipeline(file_path: str, document_id: str):
    """The actual ingestion logic running in the background."""
    try:
        logger.info(f"[Background] Starting ingestion for: {file_path}")
        split_docs = load_and_split_document(file_path)
        if not split_docs:
            logger.error(f"[Background] No documents generated for {file_path}. Aborting ingestion.")
            return

        # Add documents and embeddings to Neo4jVector
        # This creates (:Chunk) nodes with text, embedding, and metadata
        # It also ensures the vector index exists based on the store's config
        logger.info(f"[Background] Adding {len(split_docs)} chunks to vector store...")
        added_ids = neo4j_vector_store.add_documents(split_docs, ids=[d.metadata["id"] for d in split_docs])
        logger.info(f"[Background] Added {len(added_ids)} chunks to vector store.")

        if not added_ids:
             logger.warning("[Background] No chunks were successfully added to the vector store.")
             # Potentially remove the source document node if created?
             return

        # Add Document node and CONTAINS relationships
        logger.info(f"[Background] Linking chunks to document node: {document_id}")
        neo4j_graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename, d.last_updated = timestamp()
            WITH d
            MATCH (c:Chunk) WHERE c.id IN $chunk_ids
            MERGE (d)-[:CONTAINS]->(c)
        """, params={"doc_id": document_id, "filename": os.path.basename(file_path), "chunk_ids": added_ids})


        # Add NEXT_CHUNK relationships (requires chunk_index metadata)
        # Sort docs by index before creating relationships
        sorted_docs = sorted([doc for doc in split_docs if doc.metadata["id"] in added_ids], key=lambda d: d.metadata["chunk_index"])
        logger.info("[Background] Adding NEXT_CHUNK relationships...")
        for i in range(len(sorted_docs) - 1):
            prev_id = sorted_docs[i].metadata["id"]
            curr_id = sorted_docs[i+1].metadata["id"]
            neo4j_graph.query("""
                MATCH (prev:Chunk {id: $prev_id})
                MATCH (curr:Chunk {id: $curr_id})
                MERGE (prev)-[:NEXT_CHUNK]->(curr)
            """, params={"prev_id": prev_id, "curr_id": curr_id})

        logger.info(f"[Background] Successfully completed ingestion for: {file_path}")

        # Optional: Implement Entity Extraction here
        # 1. Iterate through chunks (sorted_docs)
        # 2. Call an LLM or NLP model to extract entities/relationships
        # 3. Use neo4j_graph.query() to MERGE entities and relationships

    except Exception as e:
        logger.exception(f"[Background] Error during ingestion pipeline for {file_path}: {e}")
        # Add monitoring/alerting here for production


# --- API Endpoints ---
@app.post("/upload-context")
async def upload_context(background_tasks: BackgroundTasks, file: UploadFile):
    sanitized_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    document_id = "doc_" + sanitized_filename.replace(".", "_").replace(" ", "_")

    # Save file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {sanitized_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
         if hasattr(file, 'file') and hasattr(file.file, 'close'):
             file.file.close()

    # --- Schedule ingestion as background task ---
    background_tasks.add_task(ingest_pipeline, file_path, document_id)

    return {
        "status": "processing",
        "message": f"File '{sanitized_filename}' received and scheduled for ingestion.",
        "document_id": document_id
    }


@app.post("/chat")
async def chat(message: str = Form(...)):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Received chat message (truncated): {message[:100]}...")

    try:
        # --- Define RAG Chain using LCEL ---
        vector_retriever = neo4j_vector_store.as_retriever(search_kwargs={'k': TOP_K_INITIAL_SEARCH})
        graph_enhanced_retriever = get_graph_enhanced_retriever(vector_retriever, neo4j_graph)

        # Define prompt template
        template = """You are a helpful AI assistant. Answer the user's question based *only* on the provided context. If the context does not contain the answer, state that you cannot answer based on the information available. Do not make information up. Be concise and accurate.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Define the full RAG chain
        rag_chain = (
            # RunnableParallel allows fetching context and passing question through simultaneously
            {"context": graph_enhanced_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | chat_model # Use the custom chat model wrapper
            | StrOutputParser() # Parse the output message content
        )

        # --- Invoke the RAG chain ---
        logger.info("Invoking RAG chain...")
        reply = await rag_chain.ainvoke(message) # Use async invoke for FastAPI
        # If using synchronous components wrapped:
        # reply = await asyncio.to_thread(rag_chain.invoke, message)

        logger.info("RAG chain finished, returning reply.")
        return {"reply": reply}

    except ValueError as e:
        # Specific error like query embedding failure
        logger.error(f"ValueError during chat processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        logger.error(f"LLM ConnectionError during chat processing: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to language model: {e}")
    except TimeoutError as e:
        logger.error(f"LLM TimeoutError during chat processing: {e}")
        raise HTTPException(status_code=504, detail=f"Language model request timed out: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Use reload for development