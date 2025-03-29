"""
Console UI components for the SQL RAG System.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class ConsoleUI:
    """Console user interface for the SQL RAG System."""

    def __init__(self):
        """Initialize the console UI."""
        self.console = Console()

    def show_welcome(self):
        """Display welcome message."""
        self.console.print(Panel(
            "[bold green]Comprehensive SQL RAG System for AdventureWorks[/bold green]\n\n"
            "Ask questions about sales, products, customers, and more.\n"
            "Type 'exit' to quit."
        ))

    def show_processing(self, message="Processing..."):
        """
        Display processing message.

        Args:
            message (str): Processing message
        """
        self.console.print(Panel(message, title="Processing"))

    def show_sql_query(self, sql_query):
        """
        Display generated SQL query.

        Args:
            sql_query (str): Generated SQL query
        """
        self.console.print(Panel(f"```sql\n{sql_query}\n```", title="Generated SQL Query"))

    def show_answer(self, answer):
        """
        Display answer to user query.

        Args:
            answer (str): Generated answer
        """
        self.console.print(Panel(answer, title="Answer"))

    def show_error(self, error_message):
        """
        Display error message.

        Args:
            error_message (str): Error message
        """
        self.console.print(Panel(error_message, title="[bold red]Error[/bold red]"))

    def show_prediction(self, prediction_text):
        """
        Display prediction.

        Args:
            prediction_text (str): Prediction text
        """
        self.console.print(Panel(prediction_text, title="[bold purple]AI Prediction[/bold purple]"))

    def get_user_input(self):
        """
        Get user input.

        Returns:
            str: User input
        """
        return self.console.input("[bold blue]Ask a question:[/bold blue] ")

    def show_table(self, table):
        """
        Display a Rich table.

        Args:
            table (Table): Rich table to display
        """
        if table:
            self.console.print(table)
        else:
            self.console.print("[yellow]No data to display in table.[/yellow]")