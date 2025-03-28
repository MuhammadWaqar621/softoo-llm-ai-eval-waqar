import logging
import chromadb
from typing import List, Dict, Optional
import numpy as np

from config import *

class VectorDBManager:
    """
    Manages vector database operations using Chroma.
    """
    
    def __init__(
        self, 
        collection_name: str = Config.VECTOR_DB_NAME
    ):
        """
        Initialize Chroma client and create/get collection.
        
        Args:
            collection_name (str): Name of the vector collection
        """
        try:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name=collection_name, 
                get_or_create=True
            )
            self.logger = logging.getLogger(self.__class__.__name__)
            logging.basicConfig(level=Config.LOG_LEVEL)
        except Exception as e:
            logging.error(f"Vector DB initialization failed: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Dict[str, str]], 
        embeddings: List[np.ndarray]
    ):
        """
        Add documents and their embeddings to the vector database.
        
        Args:
            documents (List[Dict[str, str]]): List of documents
            embeddings (List[np.ndarray]): Corresponding embeddings
        """
        try:
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                self.collection.add(
                    embeddings=embedding.tolist(),
                    documents=doc['text'],
                    metadatas={
                        'filename': doc['filename'],
                        'language': doc['language'],
                        'path': doc['path']
                    },
                    ids=[f"doc_{idx}"]
                )
            
            self.logger.info(f"Added {len(documents)} documents to vector DB")
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector DB: {e}")
    
    def query_documents(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Dict[str, str]]:
        """
        Query documents using an embedding.
        
        Args:
            query_embedding (np.ndarray): Embedding of the query
            top_k (int): Number of top results to return
        
        Returns:
            List[Dict[str, str]]: Top matching documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            return [
                {
                    'text': doc,
                    'metadata': metadata
                } 
                for doc, metadata in zip(
                    results['documents'][0], 
                    results['metadatas'][0]
                )
            ]
        except Exception as e:
            self.logger.error(f"Document query failed: {e}")
            return []