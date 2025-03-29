"""
Main SQL Database RAG system class.
"""
import pandas as pd
from config import (
    DB_CONNECTION_STRING, 
    GROQ_API_KEY, 
    MAX_CONVERSATION_HISTORY
)
from connection import DatabaseConnection
from schema_extractor import SchemaExtractor
from sql_generator import SQLGenerator
from response_generator import ResponseGenerator
from prediction import SalesPrediction
from relationship import RelationshipAnalyzer
from console_ui import ConsoleUI
from formatting import format_sql_result, dataframe_to_rich_table


class SQLDatabaseRAG:
    """SQL Database RAG system for natural language queries on databases."""

    def __init__(self, connection_string=DB_CONNECTION_STRING, groq_api_key=GROQ_API_KEY):
        """
        Initialize the SQL Database RAG system.

        Args:
            connection_string (str): Database connection string
            groq_api_key (str): API key for Groq
        """
        # Initialize database connection
        self.db_connection = DatabaseConnection(connection_string)
        
        # Extract schema information
        self.schema_extractor = SchemaExtractor(self.db_connection)
        self.schema_info = self.schema_extractor.extract_schema()
        
        # Initialize NLP components
        self.sql_generator = SQLGenerator(groq_api_key)
        self.response_generator = ResponseGenerator(groq_api_key)
        
        # Initialize analysis components
        self.sales_prediction = SalesPrediction(self.db_connection, groq_api_key)
        self.relationship_analyzer = RelationshipAnalyzer(self.db_connection)
        
        # Initialize UI
        self.ui = ConsoleUI()
        
        # Conversation history
        self.conversation_history = []

    def process_query(self, user_query):
        """
        Process a user query end-to-end.
        
        Args:
            user_query (str): User's natural language query
            
        Returns:
            dict: Processing results
        """
        # Store user query in conversation history
        self.conversation_history.append(("User", user_query))
        
        # Step 1: Generate SQL
        self.ui.show_processing("🔍 Generating SQL query...")
        sql_query = self.sql_generator.generate_sql(
            user_query, 
            self.schema_info, 
            self.conversation_history[-MAX_CONVERSATION_HISTORY:]
        )
        self.ui.show_sql_query(sql_query)
        
        # Step 2: Execute SQL
        self.ui.show_processing("⚙️ Executing query...")
        result = self.db_connection.execute_query(sql_query)
        
        # Convert result to DataFrame if it's a result set
        if hasattr(result, 'fetchall'):
            df = pd.DataFrame(result.fetchall())
            if not df.empty:
                df.columns = result.keys()
            result = df
        
        # Step 3: Generate response
        self.ui.show_processing("🤖 Generating response...")
        response = self.response_generator.generate_response(user_query, sql_query, result)
        
        # Step 4: Update conversation history
        self.conversation_history.append(("Assistant", response))
        
        return {
            "user_query": user_query,
            "sql_query": sql_query,
            "sql_result": result,
            "response": response
        }
    
    def handle_prediction_query(self, user_query):
        """
        Handle prediction queries.
        
        Args:
            user_query (str): User's prediction query
            
        Returns:
            str: Prediction response
        """
        self.ui.show_processing("🔮 Generating sales prediction...")
        
        # Extract product category if mentioned
        product_category = None
        if "category" in user_query.lower():
            # Simple extraction - in real system would use entity extraction
            category_words = ["bikes", "components", "clothing", "accessories"]
            for category in category_words:
                if category in user_query.lower():
                    product_category = category.capitalize()
                    break
        
        # Generate prediction
        prediction_result = self.sales_prediction.predict_future_sales(product_category)
        
        # Generate explanation
        if isinstance(prediction_result, dict):
            explanation = self.sales_prediction.generate_prediction_explanation(
                prediction_result, 
                product_category
            )
            return explanation
        else:
            return prediction_result
    
    def run_interactive_session(self):
        """Run an interactive session in the console."""
        self.ui.show_welcome()
        
        while True:
            user_query = self.ui.get_user_input()
            
            if user_query.lower() in ('exit', 'quit'):
                break
                
            # Handle different query types
            if "predict" in user_query.lower() or "forecast" in user_query.lower():
                # Handle prediction queries
                response = self.handle_prediction_query(user_query)
                self.ui.show_prediction(response)
            elif "relationship" in user_query.lower() or "schema" in user_query.lower():
                # Handle schema/relationship queries
                if "relationship" in user_query.lower():
                    result = self.relationship_analyzer.get_table_relationships()
                else:
                    result = self.relationship_analyzer.analyze_schema()
                
                if isinstance(result, pd.DataFrame):
                    table = dataframe_to_rich_table(result, "Schema Analysis")
                    self.ui.show_table(table)
                else:
                    self.ui.show_answer(result)
            else:
                # Handle regular queries
                result = self.process_query(user_query)
                self.ui.show_answer(result["response"])