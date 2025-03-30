"""
Sales prediction functionality using machine learning models for AdventureWorksLT.
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from groq import Groq
from datetime import datetime, timedelta
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS_RESPONSE
from analysis.ml_models import ModelManager
from utils.evaluation import ModelEvaluator


class SalesPrediction:
    """Predicts future sales based on historical data using ML models."""

    def __init__(self, db_connection, groq_api_key):
        """
        Initialize sales prediction.

        Args:
            db_connection: DatabaseConnection instance
            groq_api_key (str): API key for Groq
        """
        self.db_connection = db_connection
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = DEFAULT_MODEL
        self.temperature = DEFAULT_TEMPERATURE
        self.model_manager = ModelManager()
        self.evaluator = ModelEvaluator()

    def get_historical_data(self, product_category=None):
        """
        Get historical sales data from the AdventureWorksLT database.

        Args:
            product_category (str, optional): Specific product category to filter

        Returns:
            DataFrame: Historical sales data
        """
        if not self.db_connection.is_connected():
            return "Database connection required for prediction."

        # SQL to get historical sales data by month and category
        if product_category:
            # For a specific product category
            sql_query = f"""
            SELECT 
                CAST(soh.OrderDate AS DATE) AS SalesDate,
                YEAR(soh.OrderDate) AS Year,
                MONTH(soh.OrderDate) AS Month,
                DAY(soh.OrderDate) AS Day,
                pc.Name AS Category,
                SUM(sod.LineTotal) AS Sales,
                SUM(sod.OrderQty) AS Quantity,
                COUNT(DISTINCT soh.SalesOrderID) AS OrderCount
            FROM 
                SalesLT.SalesOrderHeader soh
                JOIN SalesLT.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
                JOIN SalesLT.Product p ON sod.ProductID = p.ProductID
                JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
            WHERE 
                pc.Name = '{product_category}'
            GROUP BY 
                CAST(soh.OrderDate AS DATE),
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate),
                DAY(soh.OrderDate),
                pc.Name
            ORDER BY 
                SalesDate
            """
        else:
            # For all product categories
            sql_query = """
            SELECT 
                CAST(soh.OrderDate AS DATE) AS SalesDate,
                YEAR(soh.OrderDate) AS Year,
                MONTH(soh.OrderDate) AS Month,
                DAY(soh.OrderDate) AS Day,
                pc.Name AS Category,
                SUM(sod.LineTotal) AS Sales,
                SUM(sod.OrderQty) AS Quantity,
                COUNT(DISTINCT soh.SalesOrderID) AS OrderCount
            FROM 
                SalesLT.SalesOrderHeader soh
                JOIN SalesLT.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
                JOIN SalesLT.Product p ON sod.ProductID = p.ProductID
                JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
            GROUP BY 
                CAST(soh.OrderDate AS DATE),
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate),
                DAY(soh.OrderDate),
                pc.Name
            ORDER BY 
                SalesDate
            """

        try:
            # Execute the query
            result = self.db_connection.execute_query(sql_query)
            
            if isinstance(result, str):
                # Error occurred
                return result
                
            # Convert to DataFrame
            historical_sales = pd.DataFrame(result.fetchall())
            if historical_sales.empty:
                # If no data yet, create dummy data for demonstration
                return self._generate_demo_data(product_category)
                
            historical_sales.columns = result.keys()
            # Ensure date column is datetime
            historical_sales["SalesDate"] = pd.to_datetime(historical_sales["SalesDate"])
            # Rename columns to match model expectations
            historical_sales = historical_sales.rename(columns={"SalesDate": "Date"})
            return historical_sales
            
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _generate_demo_data(self, product_category=None):
        """
        Generate demo data for forecasting if no historical data exists.
        This is only for demonstration purposes.
        
        Args:
            product_category (str, optional): Product category
            
        Returns:
            DataFrame: Generated demo data
        """
        # Starting date (one year ago)
        start_date = datetime.now() - timedelta(days=365)
        
        # Generate date range
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        # Default categories
        categories = ["Mountain Bikes", "Road Bikes", "Clothing", "Accessories"]
        if product_category:
            categories = [product_category]
        
        data = []
        for date in dates:
            for category in categories:
                # Base sales
                base_sales = {
                    "Mountain Bikes": 5000,
                    "Road Bikes": 4000,
                    "Clothing": 2000,
                    "Accessories": 1000
                }.get(category, 3000)
                
                # Add seasonal pattern (higher in summer)
                month = date.month
                seasonality = 1.0 + 0.3 * np.sin((month - 1) * np.pi / 6)
                
                # Add weekly pattern (higher on weekends)
                day_of_week = date.weekday()
                weekday_factor = 1.0 + 0.2 * (1 if day_of_week >= 5 else 0)
                
                # Add random noise
                noise = np.random.normal(1, 0.1)
                
                # Calculate sales
                sales = base_sales * seasonality * weekday_factor * noise
                
                # Add data point
                data.append({
                    "Date": date,
                    "Year": date.year,
                    "Month": date.month,
                    "Day": date.day,
                    "Category": category,
                    "Sales": round(sales, 2),
                    "Quantity": int(sales / 100),
                    "OrderCount": int(sales / 500)
                })
        
        return pd.DataFrame(data)

    def predict_future_sales(self, product_category=None, prediction_months=3, model_type="random_forest"):
        """
        Make sales predictions based on historical data using ML models.

        Args:
            product_category (str, optional): Specific product category to predict
            prediction_months (int, optional): Number of months to predict
            model_type (str, optional): Type of model to use

        Returns:
            dict: Historical and predicted sales data
        """
        # Get historical data
        historical_data = self.get_historical_data(product_category)
        
        if isinstance(historical_data, str):
            # Error occurred
            return historical_data
            
        if historical_data.empty:
            return "No historical data available for prediction."
            
        # If multiple categories, group by category
        if product_category is None:
            # Get all unique categories
            categories = historical_data["Category"].unique()
            
            # Make predictions for each category
            all_predictions = []
            
            for category in categories:
                category_data = historical_data[historical_data["Category"] == category]
                result = self._predict_single_category(category, category_data, prediction_months, model_type)
                
                if isinstance(result, dict) and "predictions" in result:
                    result["predictions"]["Category"] = category
                    all_predictions.append(result["predictions"])
            
            if all_predictions:
                # Combine predictions from all categories
                combined_predictions = pd.concat(all_predictions)
                return {"historical": historical_data, "predictions": combined_predictions}
            else:
                return "Failed to generate predictions for any category."
        else:
            # Make prediction for a single category
            return self._predict_single_category(product_category, historical_data, prediction_months, model_type)

    def _predict_single_category(self, category, historical_data, prediction_months, model_type):
        """
        Predict sales for a single category.

        Args:
            category (str): Product category
            historical_data (DataFrame): Historical data for the category
            prediction_months (int): Number of months to predict
            model_type (str): Type of model to use

        Returns:
            dict: Historical and predicted sales data
        """
        # Check if model exists
        model = self.model_manager.load_model(category, model_type)
        
        if model is None:
            # Simple time series forecasting using historical patterns if no model is available
            last_date = historical_data["Date"].max()
            
            # Generate future dates (daily)
            future_dates = []
            for i in range(1, prediction_months * 30 + 1):
                future_dates.append(last_date + timedelta(days=i))
                
            # Calculate average growth rate
            historical_data_monthly = historical_data.groupby(
                [historical_data["Year"], historical_data["Month"]]
            ).agg({"Sales": "sum"}).reset_index()
            
            historical_data_monthly["Growth"] = historical_data_monthly["Sales"].pct_change()
            avg_growth = historical_data_monthly["Growth"].mean()
            if pd.isna(avg_growth) or avg_growth < -0.5 or avg_growth > 0.5:
                avg_growth = 0.03  # Default to 3% growth if unreasonable
                
            # Get last month's sales
            last_month_sales = historical_data_monthly["Sales"].iloc[-1]
            
            # Generate predictions
            predictions = []
            current_sales = last_month_sales
            
            for i in range(1, prediction_months + 1):
                next_month = last_date.month + i
                next_year = last_date.year + (next_month - 1) // 12
                next_month = ((next_month - 1) % 12) + 1
                
                next_sales = current_sales * (1 + avg_growth)
                predictions.append({
                    "Date": datetime(next_year, next_month, 1),
                    "Year": next_year,
                    "Month": next_month,
                    "prediction": next_sales
                })
                current_sales = next_sales
            
            predictions_df = pd.DataFrame(predictions)
            return {"historical": historical_data, "predictions": predictions_df}
        else:
            # Use ML model for prediction
            # Generate future dates (daily)
            last_date = historical_data["Date"].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_months * 30 + 1)]
            
            # Make predictions
            predictions_df = model.predict(future_dates, historical_data)
            
            # Calculate prediction metrics on historical data
            X, y_true = model.prepare_features(historical_data)
            if model.model_type in ["random_forest", "linear_regression"]:
                y_pred = model.model.predict(X)
                metrics = self.evaluator.evaluate_prediction(y_true, y_pred)
                
                # Track model performance
                self.evaluator.track_model_performance(category, model_type, metrics)
                
                # Generate detailed report
                report = self.evaluator.generate_forecast_report(
                    category, model_type, historical_data, predictions_df, metrics
                )
            else:
                metrics = {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "r2": 0.0
                }
            
            return {"historical": historical_data, "predictions": predictions_df, "metrics": metrics}

    def generate_prediction_explanation(self, prediction_result, product_category=None):
        """
        Generate an explanation of the sales prediction.

        Args:
            prediction_result (dict): Prediction results
            product_category (str, optional): Specific product category

        Returns:
            str: Explanation of the prediction
        """
        if isinstance(prediction_result, dict):
            # Format prediction result for display
            prediction_text = "Historical sales data:\n"
            if "historical" in prediction_result:
                # Group by month for display
                historical_monthly = prediction_result["historical"].groupby(
                    [prediction_result["historical"]["Year"], prediction_result["historical"]["Month"]]
                ).agg({"Sales": "sum"}).tail(5).reset_index()
                prediction_text += historical_monthly.to_string() + "\n\n"
            
            prediction_text += "Sales predictions for the next few months:\n"
            if "predictions" in prediction_result:
                prediction_text += prediction_result["predictions"].to_string() + "\n\n"
            
            # Add metrics if available
            if "metrics" in prediction_result:
                prediction_text += "Model performance metrics:\n"
                prediction_text += f"RMSE: {prediction_result['metrics']['rmse']:.2f}\n"
                prediction_text += f"R²: {prediction_result['metrics']['r2']:.4f}\n"

            # Generate explanation with Groq
            prompt = f"""You are an AI assistant that explains sales predictions.
Sales prediction data:
{prediction_text}
Product Category: {product_category if product_category else "All categories"}
Provide a detailed explanation of these sales predictions, including:
1. The overall trend and expected growth rate
2. Any seasonal patterns or cyclical behavior
3. Key insights for business decision-making
4. Confidence level of the prediction
Label your response as an AI prediction.
"""

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=MAX_TOKENS_RESPONSE
            )

            return response.choices[0].message.content
        else:
            return f"Error generating prediction explanation: {prediction_result}"

    def evaluate_predictions(self, actual_data, predicted_data):
        """
        Evaluate prediction accuracy against actual data.

        Args:
            actual_data (DataFrame): Actual sales data
            predicted_data (DataFrame): Predicted sales data

        Returns:
            dict: Evaluation metrics
        """
        # Merge actual and predicted data on date
        merged_data = pd.merge(
            actual_data, 
            predicted_data,
            on="Date",
            how="inner",
            suffixes=("_actual", "_pred")
        )
        
        if merged_data.empty:
            return "No matching dates between actual and predicted data."
        
        # Calculate metrics
        y_true = merged_data["Sales"]
        y_pred = merged_data["prediction"]
        
        metrics = self.evaluator.evaluate_prediction(y_true, y_pred)
        
        return metrics

    def get_model_metrics_history(self, category, model_type):
        """
        Get the performance history of a specific model.

        Args:
            category (str): Product category
            model_type (str): Model type

        Returns:
            DataFrame: Performance history
        """
        return self.evaluator.get_model_history(category, model_type)