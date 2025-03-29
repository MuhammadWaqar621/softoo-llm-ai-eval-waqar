"""
Configuration settings for the SQL RAG System.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Database connection string
DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "mssql+pyodbc://@localhost/AdventureWorksLT2022?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

# LLM model settings
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.1
MAX_TOKENS_SQL = 500
MAX_TOKENS_RESPONSE = 1000

# Memory settings
MAX_CONVERSATION_HISTORY = 5