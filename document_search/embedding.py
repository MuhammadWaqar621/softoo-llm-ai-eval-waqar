"""
Embedding generation module for Multilingual RAG System.
"""
from typing import Any
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config
from langchain_core.embeddings import Embeddings  
class EmbeddingManager:
    """
    Manager for generating document embeddings with multilingual support.
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Optional model name to override the default
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._embedding_model = None
        print(f"📊 Initializing embedding model: {self.model_name}")
        
    @property
    def embedding_model(self) -> Embeddings:
        """Get the embedding model, initializing it if necessary."""
        if self._embedding_model is None:
            self._initialize_model()
        return self._embedding_model
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            # Load the embedding model
            model_kwargs = {}
            
            # Check for GPU availability
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_kwargs['device'] = device
                print(f"🖥️ Using device: {device}")
            except ImportError:
                print("⚠️ PyTorch not available, using CPU")
            
            self._embedding_model = SentenceTransformerEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs
            )
            print(f"✅ Successfully loaded embedding model: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {str(e)}")
            print(f"⚠️ Falling back to simpler model: {config.EMBEDDING_MODEL_MINILM}")
            self._embedding_model = SentenceTransformerEmbeddings(
                model_name=config.EMBEDDING_MODEL_MINILM
            )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Test on a simple string
        test_embedding = self.embedding_model.embed_query("Test string")
        return len(test_embedding)
    
    @staticmethod
    def recommend_model_for_languages(languages):
        """
        Recommend the best embedding model based on languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            Recommended model name
        """
        # If only English, use the faster model
        if languages == ["en"]:
            return config.EMBEDDING_MODEL_MINILM
            
        # If it includes languages with non-Latin scripts (Arabic, Urdu, etc.)
        non_latin = any(lang in ["ar", "ur", "zh", "ja", "ko", "ru", "hi"] for lang in languages)
        if non_latin:
            return config.EMBEDDING_MODEL_MULTILINGUAL_MPNET
            
        # Default to the balanced multilingual model
        return config.EMBEDDING_MODEL_MULTILINGUAL_MINILM