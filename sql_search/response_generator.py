"""
Natural language response generation from SQL results.
"""
import pandas as pd
from groq import Groq
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS_RESPONSE


class ResponseGenerator:
    """Generates natural language responses from SQL query results."""

    def __init__(self, groq_api_key):
        """
        Initialize response generator.

        Args:
            groq_api_key (str): API key for Groq
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = DEFAULT_MODEL
        self.temperature = DEFAULT_TEMPERATURE

    def generate_response(self, user_query, sql_query, sql_result):
        """
        Generate a natural language response based on the SQL results.

        Args:
            user_query (str): User's natural language query
            sql_query (str): Executed SQL query
            sql_result: Result of the SQL query execution

        Returns:
            str: Natural language response
        """
        # Format SQL results for the LLM
        if isinstance(sql_result, pd.DataFrame):
            if sql_result.empty:
                result_text = "The query returned no results."
            else:
                # Convert to string with reasonable formatting
                result_text = f"The query returned {len(sql_result)} rows. Here is a sample:\n"
                result_text += sql_result.head(10).to_string()
        else:
            result_text = str(sql_result)

        prompt = f"""You are an AI assistant that helps users understand data from the AdventureWorks database.
User query: "{user_query}"
SQL query executed:
```sql
{sql_query}
```
Query results:
{result_text}
Based on the SQL query and its results, provide a clear and concise answer to the user's question. 
If the results contain meaningful insights, mention them. 
If the query returned an error or no results, suggest a better approach if possible.
"""

        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=MAX_TOKENS_RESPONSE
        )

        return response.choices[0].message.content