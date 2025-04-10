
"""
Configuration settings and constants for the Multilingual RAG System.
"""

# File and directory paths
DOCUMENTS_DIRECTORY = "data"
PERSIST_DIRECTORY = "vector_db"
LOG_DIRECTORY = "logs"
TEMP_DIRECTORY = "temp_processing"

# Embedding models
# General purpose
EMBEDDING_MODEL_MINILM = "all-MiniLM-L6-v2"  # Fast and balanced
# Multilingual models (better for non-English content)
EMBEDDING_MODEL_MULTILINGUAL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"  # Good multilingual support
EMBEDDING_MODEL_MULTILINGUAL_MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Best multilingual quality

# Default embedding model - change this based on your language needs
# EMBEDDING_MODEL = EMBEDDING_MODEL_MINILM  # For English-only content
EMBEDDING_MODEL = EMBEDDING_MODEL_MULTILINGUAL_MINILM  # For multilingual content

# Document chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# LLM settings
LLM_PROVIDER = "groq"  # Options: "groq", "openai", etc.
LLM_MODEL = "llama3-8b-8192"  # Groq model
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 500

# Retrieval settings
TOP_K_CHUNKS = 5

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
# groq apikey
APIKEY="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"
# Language support - helpful for optimization based on your content
LANGUAGES = ["en", "ar", "ur", "hi"]  



"""
Configuration settings for the SQL RAG System.
"""
# Groq API key
GROQ_API_KEY = "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"

# Database connection string
DB_CONNECTION_STRING = "mssql+pyodbc://@localhost/AdventureWorksLT2022?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
# LLM model settings
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.1
MAX_TOKENS_SQL = 500
MAX_TOKENS_RESPONSE = 1000

# Memory settings
MAX_CONVERSATION_HISTORY = 5

# Model directories
MODELS_DIR = "models"
RESULTS_DIR = "results"