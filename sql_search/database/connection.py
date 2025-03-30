"""
Database connection handling for the SQL RAG System.
"""
import pandas as pd
import sqlalchemy as db
from sqlalchemy.exc import SQLAlchemyError
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
            Result of the query execution or error message
        """
        if not self.engine:
            return "Database connection not available."

        try:
            # Clean the query to remove any potential backticks
            query = query.replace('`', '')
            
            # Execute the query
            with self.engine.connect() as conn:
                result = conn.execute(db.text(query))
                
                # Try to get column names - this will fail for non-SELECT queries
                try:
                    column_names = result.keys()
                    # This is a SELECT query with results
                    rows = result.fetchall()
                    # Create a DataFrame with the result
                    df = pd.DataFrame(rows, columns=column_names)
                    return df
                except Exception:
                    # This might be a query without results (INSERT, UPDATE, etc.)
                    # Or a SELECT that returns a scalar value
                    conn.commit()  # Make sure to commit any changes
                    
                    # Try to get a scalar result
                    try:
                        # For scalar queries like "SELECT COUNT(*)"
                        scalar_result = conn.scalar(db.text(query))
                        return pd.DataFrame({'Result': [scalar_result]})
                    except Exception:
                        # For queries that don't return anything
                        return pd.DataFrame({'Result': ['Query executed successfully']})
                        
        except SQLAlchemyError as e:
            error_msg = str(e.__dict__.get('orig', e))
            return f"Error executing SQL query: {error_msg}"
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"

    def is_connected(self):
        """
        Check if database connection is established.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.engine is not None