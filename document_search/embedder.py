# document_search/src/embedder.py

import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config

class DocumentEmbedder:
    """
    A class to generate embeddings for documents using sentence transformers.
    """
    
    def __init__(
        self, 
        model_name: str = Config.EMBEDDING_MODEL
    ):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name (str): Name of the embedding model
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.logger = logging.getLogger(self.__class__.__name__)
            logging.basicConfig(level=Config.LOG_LEVEL)
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(
        self, 
        documents: List[Dict[str, str]]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents (List[Dict[str, str]]): List of document dictionaries
        
        Returns:
            List[np.ndarray]: List of embeddings
        """
        try:
            texts = [doc['text'] for doc in documents]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
            texts = text_splitter.split_documents(documents)
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
    
    def semantic_search(
        self, 
        query: str, 
        documents: List[Dict[str, str]], 
        embeddings: List[np.ndarray],
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Dict[str, str]]:
        """
        Perform semantic search on documents.
        
        Args:
            query (str): Search query
            documents (List[Dict[str, str]]): List of documents
            embeddings (List[np.ndarray]): Pre-computed embeddings
            top_k (int): Number of top results to return
        
        Returns:
            List[Dict[str, str]]: Top matching documents
        """
        try:
            query_embedding = self.model.encode(query)
            
            # Compute cosine similarities
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * 
                np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [documents[idx] for idx in top_indices]
        
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []