from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from config import (
    GRAPH_CONTEXT_NEIGHBORS,
    logger)
def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for the LLM context."""
    # Simple join, consider adding metadata like source/page if useful
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_graph_enhanced_retriever(vector_retriever: BaseRetriever, graph: Neo4jGraph, k_neighbors: int = GRAPH_CONTEXT_NEIGHBORS) -> BaseRetriever:
    """
    A retriever that combines vector search with graph neighborhood traversal.
    """
    def fetch_neighbors(docs: List[Document]) -> List[Document]:
        """Fetches neighboring chunks from the graph for the initially retrieved docs."""
        if not docs or k_neighbors <= 0:
            return docs # Return original docs if no neighbors requested or no initial docs

        all_docs_map = {doc.metadata["id"]: doc for doc in docs}
        neighbor_query = """
        MATCH (c:Chunk) WHERE c.id IN $chunk_ids
        // Fetch previous and next neighbors up to k_neighbors distance
        CALL {
            WITH c
            MATCH path = (prev:Chunk)-[:NEXT_CHUNK*1..2]->(c)
            RETURN prev as neighbor, length(path) as dist
            ORDER BY dist DESC // Closest previous neighbor first
        UNION
            WITH c
            MATCH path = (c)-[:NEXT_CHUNK*1..2]->(next:Chunk)
            RETURN next as neighbor, length(path) as dist
            ORDER BY dist ASC // Closest next neighbor first
        }
        WITH neighbor WHERE NOT neighbor.id IN $chunk_ids // Only add neighbors not already retrieved
        RETURN DISTINCT neighbor.id AS id, neighbor.text AS text, neighbor.source_document AS source_document, neighbor.chunk_index AS chunk_index
        """
        chunk_ids = list(all_docs_map.keys())
        try:
            results = graph.query(neighbor_query, params={"chunk_ids": chunk_ids, "k_neighbors": k_neighbors})
            added_neighbors = 0
            for record in results:
                neighbor_id = record["id"]
                if neighbor_id not in all_docs_map:
                    neighbor_doc = Document(
                        page_content=record["text"],
                        metadata={
                            "id": neighbor_id,
                            "source_document": record["source_document"],
                            "chunk_index": record["chunk_index"],
                            "retrieval_source": "graph_neighbor" # Mark how it was retrieved
                        }
                    )
                    all_docs_map[neighbor_id] = neighbor_doc
                    added_neighbors += 1
            logger.info(f"Fetched {added_neighbors} unique neighbor chunks from graph.")
        except Exception as e:
            logger.error(f"Failed to fetch graph neighbors: {e}")
            # Proceed with only vector results if graph traversal fails

        # Return combined list of documents (original + neighbors)
        # Optional: Re-order based on chunk_index or relevance? For now, just combine.
        return list(all_docs_map.values())

    # Create a runnable sequence
    # 1. Run vector retriever
    # 2. Pass results to fetch_neighbors
    # 3. Format the combined docs
    graph_retriever_runnable = (
        vector_retriever # Input: query string -> Output: List[Document]
        | RunnableLambda(fetch_neighbors) # Input: List[Document] -> Output: List[Document] (original + neighbors)
    )
    return graph_retriever_runnable
