from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_DIMENSION, get_logger)

logger = get_logger(__name__)

def get_neo4j_graph_instance():
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    try:
        graph.query("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        graph.query("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

        logger.info("Ensured Neo4j constraints exist.")

        # Create vector index (handled by Neo4jVector, but good to know syntax)
        # graph.query(f"""
        # CREATE VECTOR INDEX `chunk_embeddings` IF NOT EXISTS
        # FOR (c:Chunk) ON (c.embedding)
        # OPTIONS {{indexConfig: {{
        #     `vector.dimensions`: {EMBEDDING_DIMENSION},
        #     `vector.similarity_function`: 'cosine'
        # }}}}
        # """)
        # logger.info("Ensured Neo4j vector index exists/is configured.")

    except Exception as e:
        logger.warning(f"Could not ensure Neo4j constraints/indices (might require DB admin privileges or DB restart): {e}")
    return graph


def get_neo4j_vector_store(embedding_function: Embeddings):
        """
        Creates a Neo4j vector store using the provided embedding function.
        
        Args:
            embedding_function: The embedding function to use for vectorizing text
            
        Returns:
            A configured Neo4jVector instance connected to the database
        
        Note:
            Vector index creation is handled automatically by Neo4jVector
        """
        return Neo4jVector.from_existing_graph(
            embedding=embedding_function,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="chunk_embeddings", 
            node_label="Chunk",            # Label for nodes with embeddings
            text_node_properties=["text"], # Property storing the text
            embedding_node_property="embedding", # Property storing the embedding
            # Define how documents map to graph properties
            # Often includes 'id' if you explicitly set it, or Neo4j's internal ID
            retrieval_query="""
            RETURN node.text AS text, score, {
                id: node.id,
                source: node.source_document,
                chunk_index: node.chunk_index
                
            } AS metadata
            """
    )