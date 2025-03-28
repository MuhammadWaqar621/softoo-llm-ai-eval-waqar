import logging
from config import Config
from document_search.document_loader import DocumentLoader
from document_search.embedder import DocumentEmbedder
from document_search.vectordb_manager import VectorDBManager
from document_search.embedding_persistence import EmbeddingPersistenceManager

def main():
    """
    Main execution script for document search module with embedding persistence.
    """
    logging.basicConfig(level=Config.LOG_LEVEL)
    logger = logging.getLogger('DocumentSearchModule')
    
    try:
        # 1. Load Documents
        loader = DocumentLoader()
        documents = loader.load_documents()
        print(documents)
        # # 2. Manage Embeddings
        # embedder = DocumentEmbedder()
        # persistence_manager = EmbeddingPersistenceManager()
        
        # # Check for existing embeddings
        # existing_embeddings, documents_to_embed = persistence_manager.get_existing_embeddings(documents)
        
        # # Generate embeddings for new documents
        # if documents_to_embed:
        #     new_embeddings = embedder.generate_embeddings(documents_to_embed)
            
        #     # Combine existing and new embeddings
        #     persistence_manager.save_embeddings(documents_to_embed, new_embeddings)
            
        #     # Combine embeddings lists
        #     all_embeddings = existing_embeddings + list(new_embeddings)
        #     all_documents = [
        #         doc for doc in documents 
        #         if doc not in documents_to_embed
        #     ] + documents_to_embed
        # else:
        #     all_embeddings = existing_embeddings
        #     all_documents = documents
        
        # # 3. Store in Vector DB
        # vectordb = VectorDBManager()

        # vectordb.add_documents(all_documents, all_embeddings)
        
        # # Example Search
        # query = "What is artificial intelligence?"
        # query_embedding = embedder.model.encode(query)
        # search_results = vectordb.query_documents(query_embedding)
        
        # logger.info("Document Search Module Initialized Successfully")
        
        # # Print search results
        # for result in search_results:
        #     print(f"Document: {result['metadata']['filename']}")
        #     print(f"Snippet: {result['text']}...\n")
    
    except Exception as e:
        logger.error(f"Document Search Module Failed: {e}")

if __name__ == "__main__":
    main()