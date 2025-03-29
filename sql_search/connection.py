"""
Database connection handling for the SQL RAG System.
"""
import sqlalchemy as db
from rich.console import Console

console = Console()


class DatabaseConnection:
    """Handles database connection and query execution."""

    def __init__(self, connection_string):
        """
        Initialize database connection.

        Args:
            connection_string (str): Database connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.connection = None
        self._establish_connection()

    def _establish_connection(self):
        """Establish connection to the database."""
        try:
            self.engine = db.create_engine(self.connection_string)
            self.connection = self.engine.connect()
            console.print("[green]Successfully connected to the database.[/green]")
        except Exception as e:
            console.print(f"[red]Error connecting to database: {e}[/red]")
            self.engine = None
            self.connection = None

    def execute_query(self, query):
        """
        Execute SQL query and return results.

        Args:
            query (str): SQL query to execute

        Returns:
            Result of the query execution
        """
        if not self.engine:
            return "Database connection not available."

        try:
            with self.engine.connect() as conn:
                result = conn.execute(db.text(query))
                return result
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"

    def is_connected(self):
        """
        Check if database connection is established.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.engine is not None