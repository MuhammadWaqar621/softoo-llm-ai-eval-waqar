"""
SQL generation functionality using LLMs.
"""
from groq import Groq
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS_SQL


class SQLGenerator:
    """Generates SQL queries from natural language using LLMs."""

    def __init__(self, groq_api_key):
        """
        Initialize SQL generator.

        Args:
            groq_api_key (str): API key for Groq
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = DEFAULT_MODEL
        self.temperature = DEFAULT_TEMPERATURE

    def generate_sql(self, user_query, schema_info, conversation_history=None):
        """
        Generate SQL query from natural language using Groq.

        Args:
            user_query (str): User's natural language query
            schema_info (str): Database schema information
            conversation_history (list, optional): Conversation history

        Returns:
            str: Generated SQL query
        """
        # Format conversation history if provided
        history_text = self._format_conversation_history(conversation_history) if conversation_history else "No previous conversation."

        # Construct prompt with schema information and user query
        prompt = f"""You are an AI assistant that converts natural language questions about the AdventureWorks database into SQL queries.
        
Here is the database schema information:
{schema_info}

Recent conversation history:
{history_text}

Based on the schema above and the conversation history, write a SQL query to answer this question: "{user_query}"
Return ONLY the SQL query without any explanation. The query should be correct, executable SQL for SQL Server.
"""

        # Get SQL query from Groq
        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=MAX_TOKENS_SQL
        )

        # Extract SQL query
        sql_query = response.choices[0].message.content.strip()

        # Remove any markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        return sql_query

    def _format_conversation_history(self, conversation_history):
        """
        Format conversation history for context.

        Args:
            conversation_history (list): List of (role, message) tuples

        Returns:
            str: Formatted conversation history
        """
        if not conversation_history:
            return "No previous conversation."

        formatted_history = ""
        for role, message in conversation_history:
            formatted_history += f"{role}: {message}\n"

        return formatted_history