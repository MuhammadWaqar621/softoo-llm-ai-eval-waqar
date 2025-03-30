"""
Formatting utilities for the SQL RAG System.
"""
import pandas as pd
from rich.table import Table
from rich.console import Console

console = Console()


def format_conversation_history(conversation_history, max_history=5):
    """
    Format conversation history for display.

    Args:
        conversation_history (list): List of (role, message) tuples
        max_history (int, optional): Maximum number of history items to include

    Returns:
        str: Formatted conversation history
    """
    if not conversation_history:
        return "No previous conversation."
    
    formatted_history = ""
    for role, message in conversation_history[-max_history:]:
        formatted_history += f"{role}: {message}\n"
    
    return formatted_history


def dataframe_to_rich_table(df, title=None):
    """
    Convert a DataFrame to a Rich table for console display.

    Args:
        df (DataFrame): DataFrame to convert
        title (str, optional): Table title

    Returns:
        Table: Rich table
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    table = Table(title=title, show_header=True, header_style="bold")
    
    # Add columns
    for column in df.columns:
        table.add_column(str(column))
    
    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(value) for value in row.values])
    
    return table


def format_sql_result(result, max_rows=20):
    """
    Format SQL result for display.

    Args:
        result: SQL query result
        max_rows (int, optional): Maximum number of rows to display

    Returns:
        str: Formatted result
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Query returned no results."
        
        # If DataFrame is too large, truncate it
        if len(result) > max_rows:
            truncated = result.head(max_rows)
            return f"Query returned {len(result)} rows. Showing first {max_rows}:\n{truncated.to_string()}"
        else:
            return f"Query returned {len(result)} rows:\n{result.to_string()}"
    elif isinstance(result, str):
        return result
    else:
        return str(result)