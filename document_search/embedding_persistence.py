import os
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from config import Config

class EmbeddingPersistenceManager:
    """
    Manages persistent storage and retrieval of document embeddings.
    """
    
    def __init__(
        self, 
        cache_dir: Path = Config.BASE_DIR / 'cache'
    ):
        """
        Initialize embedding persistence manager.
        
        Args:
            cache_dir (Path): Directory to store embedding cache
        """
        self.cache_dir = cache_dir
        self.embeddings_cache_file = cache_dir / 'document_embeddings.json'
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_document_hash(self, document: Dict[str, str]) -> str:
        """
        Generate a unique hash for a document based on its content.
        
        Args:
            document (Dict[str, str]): Document dictionary
        
        Returns:
            str: Unique hash for the document
        """
        # Use filename and content to generate hash
        hash_input = f"{document['filename']}_{document['text']}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def save_embeddings(
        self, 
        documents: List[Dict[str, str]], 
        embeddings: List[np.ndarray]
    ):
        """
        Save embeddings to a persistent cache.
        
        Args:
            documents (List[Dict[str, str]]): List of documents
            embeddings (List[np.ndarray]): Corresponding embeddings
        """
        # Load existing cache or create new
        cache = self._load_cache()
        
        # Update cache with new embeddings
        for doc, embedding in zip(documents, embeddings):
            doc_hash = self._generate_document_hash(doc)
            cache[doc_hash] = {
                'filename': doc['filename'],
                'embedding': embedding.tolist(),
                'language': doc['language']
            }
        
        # Save updated cache
        with open(self.embeddings_cache_file, 'w') as f:
            json.dump(cache, f)
    
    def _load_cache(self) -> Dict[str, Dict]:
        """
        Load existing embedding cache.
        
        Returns:
            Dict[str, Dict]: Cached embeddings
        """
        if os.path.exists(self.embeddings_cache_file):
            with open(self.embeddings_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_existing_embeddings(
        self, 
        documents: List[Dict[str, str]]
    ) -> tuple:
        """
        Retrieve existing embeddings for documents.
        
        Args:
            documents (List[Dict[str, str]]): List of documents
        
        Returns:
            tuple: (existing_embeddings, documents_to_embed)
        """
        cache = self._load_cache()
        existing_embeddings = []
        documents_to_embed = []
        
        for doc in documents:
            doc_hash = self._generate_document_hash(doc)
            
            if doc_hash in cache:
                # Use cached embedding
                existing_embeddings.append(np.array(cache[doc_hash]['embedding']))
            else:
                # Document needs new embedding
                documents_to_embed.append(doc)
        
        return existing_embeddings, documents_to_embed