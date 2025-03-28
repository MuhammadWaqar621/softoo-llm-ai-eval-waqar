# document_search/config/config.py

from pathlib import Path
import os

class Config:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.path.join("document_search","data"))
    
    
    # Embedding configuration
    EMBEDDING_MODEL = 'multi-qa-MiniLM-L6-cos-v1'
    
    # Vector DB configuration
    VECTOR_DB_NAME = 'document_collection'
    
    # Search configuration
    DEFAULT_TOP_K = 5
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    
    # Supported languages
    SUPPORTED_LANGUAGES = ['en', 'ar', 'ur']

