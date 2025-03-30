"""
Main entry point for the SQL Database RAG system with forecasting capabilities.
"""
import argparse
from sql_database_rag import SQLDatabaseRAG
from config.settings import DB_CONNECTION_STRING, GROQ_API_KEY
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SQL Database RAG System with Forecasting')
    parser.add_argument('--db', type=str, help='Database connection string')
    parser.add_argument('--api-key', type=str, help='Groq API key')
    parser.add_argument('--train', action='store_true', help='Train ML models before starting')
    args = parser.parse_args()
    
    # Use provided values or defaults from settings
    db_connection = args.db if args.db else DB_CONNECTION_STRING
    api_key = args.api_key if args.api_key else GROQ_API_KEY
    
    console.print(Panel(
        "[bold green]SQL Database RAG System with Forecasting[/bold green]\n\n"
        "Connecting to database and initializing system...",
        title="Starting"
    ))
    
    # Initialize the RAG system
    rag_system = SQLDatabaseRAG(db_connection, api_key)
    
    # Train models if requested
    if args.train:
        console.print(Panel("Training ML forecasting models...", title="Initialization"))
        rag_system.handle_training_query("train models")
    
    # Run interactive session
    rag_system.run_interactive_session()


if __name__ == "__main__":
    main()