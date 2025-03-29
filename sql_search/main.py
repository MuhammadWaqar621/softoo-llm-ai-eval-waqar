"""
Main entry point for the SQL Database RAG system.
"""
import argparse
from rag import SQLDatabaseRAG
from config import DB_CONNECTION_STRING, GROQ_API_KEY


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SQL Database RAG System')
    parser.add_argument('--db', type=str, help='Database connection string')
    parser.add_argument('--api-key', type=str, help='Groq API key')
    args = parser.parse_args()
    
    # Use provided values or defaults from settings
    db_connection = args.db if args.db else DB_CONNECTION_STRING
    api_key = args.api_key if args.api_key else GROQ_API_KEY
    
    # Initialize the RAG system
    rag_system = SQLDatabaseRAG(db_connection, api_key)
    
    # Run interactive session
    rag_system.run_interactive_session()


if __name__ == "__main__":
    main()