"""
Main SQL Database RAG system class with ML-based forecasting for AdventureWorksLT.
"""
import pandas as pd
from config import (
    DB_CONNECTION_STRING, 
    GROQ_API_KEY, 
    MAX_CONVERSATION_HISTORY
)
from database.connection import DatabaseConnection
from database.schema_extractor import SchemaExtractor
from nlp.sql_generator import SQLGenerator
from nlp.response_generator import ResponseGenerator
from analysis.prediction import SalesPrediction
from analysis.relationship import RelationshipAnalyzer
from analysis.model_training import ModelTrainer
from ui.console_ui import ConsoleUI
from utils.formatting import format_sql_result, dataframe_to_rich_table
from utils.evaluation import ModelEvaluator


class SQLDatabaseRAG:
    """SQL Database RAG system for natural language queries and sales forecasting."""

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
        self.model_trainer = ModelTrainer(self.db_connection)
        self.model_evaluator = ModelEvaluator()
        
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
        
        # Handle string result (error message)
        if isinstance(result, str):
            error_result = f"Error: {result}"
            # Update conversation history
            self.conversation_history.append(("Assistant", error_result))
            return {
                "user_query": user_query,
                "sql_query": sql_query,
                "sql_result": error_result,
                "response": error_result
            }
            
        # At this point, result should be a DataFrame
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame({'Result': ['Query executed but returned unexpected result type']})
        
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
        Handle prediction queries using ML models.
        
        Args:
            user_query (str): User's prediction query
            
        Returns:
            str: Prediction response
        """
        self.ui.show_processing("🔮 Generating sales prediction with ML models...")
        
        # Extract product category if mentioned
        product_category = None
        if "category" in user_query.lower():
            # Simple extraction - in real system would use entity extraction
            # Check common categories in AdventureWorksLT
            category_words = [
                "bikes", "components", "clothing", "accessories", 
                "mountain", "road", "touring", "helmets"
            ]
            for category in category_words:
                if category in user_query.lower():
                    if category == "mountain" or category == "road" or category == "touring":
                        product_category = f"{category.capitalize()} Bikes"
                    else:
                        product_category = category.capitalize()
                    break
        
        # Extract model type if mentioned
        model_type = "random_forest"  # default
        if "arima" in user_query.lower():
            model_type = "arima"
        elif "linear" in user_query.lower():
            model_type = "linear_regression"
        
        # Extract prediction period if mentioned
        prediction_months = 3  # default
        if "month" in user_query.lower() or "months" in user_query.lower():
            # Try to extract number
            import re
            matches = re.findall(r'(\d+)\s*month', user_query.lower())
            if matches:
                prediction_months = int(matches[0])
        
        # Generate prediction
        prediction_result = self.sales_prediction.predict_future_sales(
            product_category, 
            prediction_months,
            model_type
        )
        
        # Generate explanation
        if isinstance(prediction_result, dict):
            explanation = self.sales_prediction.generate_prediction_explanation(
                prediction_result, 
                product_category
            )
            return explanation
        else:
            return prediction_result
    def handle_training_query(self, user_query):
        """
        Handle model training queries.
        
        Args:
            user_query (str): User's training query
            
        Returns:
            str: Training response
        """
        self.ui.show_processing("🧠 Training ML forecasting models...")
        
        # Extract parameters
        cross_validate = "cross" in user_query.lower() and "valid" in user_query.lower()
        use_demo = "demo" in user_query.lower() or "sample" in user_query.lower()
        
        # Get training data
        if use_demo:
            self.ui.show_processing("Generating demo data...")
            training_data = self.model_trainer._generate_demo_data()
        else:
            training_data = self.model_trainer.get_training_data()
        
        if training_data is None or isinstance(training_data, str):
            return "Could not retrieve training data: " + (training_data or "Unknown error")
        
        if training_data.empty:
            return "No training data available. Please check the database or use demo data."
        
        # Train models
        if cross_validate:
            self.ui.show_processing("Performing cross-validation...")
            metrics = self.model_trainer.cross_validate(training_data)
            
            # Format metrics
            if metrics:
                metrics_text = "Cross-Validation Results:\n\n"
                any_metrics = False
                
                for category in metrics:
                    for model_type, model_metrics in metrics[category].items():
                        if model_metrics:  # Only show categories with actual metrics
                            any_metrics = True
                            metrics_text += f"Category: {category}\n"
                            metrics_text += f"  Model: {model_type}\n"
                            if "avg_rmse" in model_metrics:
                                metrics_text += f"    Average RMSE: {model_metrics['avg_rmse']:.2f}\n"
                            if "avg_r2" in model_metrics:
                                metrics_text += f"    Average R²: {model_metrics['avg_r2']:.4f}\n"
                            metrics_text += "\n"
                
                if any_metrics:
                    return metrics_text
                else:
                    return "No models could be cross-validated due to insufficient data. Try using demo data with 'train models with demo data'."
            else:
                return "Cross-validation failed or no results were produced."
        else:
            self.ui.show_processing("Training models...")
            if use_demo:
                train_metrics, test_metrics = self.model_trainer.train_on_demo_data()
            else:
                train_metrics, test_metrics = self.model_trainer.train_and_evaluate(training_data)
            
            # Show metrics table
            metrics_table = self.model_trainer.display_metrics()
            if metrics_table:
                self.ui.show_table(metrics_table)
            
            # Create summary text
            summary = "Model Training Results:\n\n"
            any_results = False
            
            for category in test_metrics or {}:
                best_model, best_metrics = self.model_trainer.find_best_model(category)
                if best_model and best_metrics and 'rmse' in best_metrics:
                    any_results = True
                    summary += f"Category: {category}\n"
                    summary += f"  Best Model: {best_model}\n"
                    summary += f"  Test RMSE: {best_metrics['rmse']:.2f}\n"
                    summary += f"  Test R²: {best_metrics['r2']:.4f}\n\n"
            
            if any_results:
                return summary
            else:
                if use_demo:
                    return "Training completed, but no models met the quality threshold. Please check the data or adjust model parameters."
                else:
                    return "No models were trained due to insufficient data. Try using demo data with 'train models with demo data'."
        
    def handle_model_query(self, user_query):
        """
        Handle model-related queries.
        
        Args:
            user_query (str): User's model query
            
        Returns:
            str: Model response
        """
        if "train" in user_query.lower():
            return self.handle_training_query(user_query)
        elif "predict" in user_query.lower() or "forecast" in user_query.lower():
            return self.handle_prediction_query(user_query)
        else:
            return "Unsure what model operation to perform. Try asking for training or prediction specifically."
    
    def run_interactive_session(self):
        """Run an interactive session in the console."""
        self.ui.show_welcome()
        
        while True:
            user_query = self.ui.get_user_input()
            
            if user_query.lower() in ('exit', 'quit'):
                break
                
            # Handle different query types
            if any(keyword in user_query.lower() for keyword in ["predict", "forecast", "train", "model"]):
                # Handle ML model queries
                response = self.handle_model_query(user_query)
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