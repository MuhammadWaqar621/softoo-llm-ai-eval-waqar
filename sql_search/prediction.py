"""
Sales prediction functionality.
"""
import pandas as pd
from sqlalchemy import text
from groq import Groq
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS_RESPONSE


class SalesPrediction:
    """Predicts future sales based on historical data."""

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

    def predict_future_sales(self, product_category=None, prediction_months=3):
        """
        Make sales predictions based on historical data.

        Args:
            product_category (str, optional): Specific product category to predict
            prediction_months (int, optional): Number of months to predict

        Returns:
            dict: Historical and predicted sales data
        """
        if not self.db_connection.is_connected():
            return "Database connection required for prediction."

        # SQL to get historical sales data by month and category
        if product_category:
            # For a specific product category
            sql_query = f"""
            SELECT 
                YEAR(soh.OrderDate) as Year,
                MONTH(soh.OrderDate) as Month,
                pc.Name as CategoryName,
                SUM(sod.OrderQty * sod.UnitPrice) as TotalSales
            FROM 
                Sales.SalesOrderHeader soh
                JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
                JOIN Production.Product p ON sod.ProductID = p.ProductID
                JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
                JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
            WHERE 
                pc.Name = '{product_category}'
            GROUP BY 
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate),
                pc.Name
            ORDER BY 
                Year, Month
            """
        else:
            # For all product categories
            sql_query = """
            SELECT 
                YEAR(soh.OrderDate) as Year,
                MONTH(soh.OrderDate) as Month,
                SUM(sod.OrderQty * sod.UnitPrice) as TotalSales
            FROM 
                Sales.SalesOrderHeader soh
                JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
            GROUP BY 
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate)
            ORDER BY 
                Year, Month
            """

        try:
            # Execute the query
            query_result = self.db_connection.execute_query(sql_query)
            
            if isinstance(query_result, str):
                # Error occurred
                return query_result
                
            # Convert to DataFrame
            historical_sales = pd.DataFrame(query_result.fetchall())
            if not historical_sales.empty:
                historical_sales.columns = query_result.keys()

                # Create time feature and sort by date
                historical_sales['Date'] = pd.to_datetime(historical_sales[['Year', 'Month']].assign(DAY=1))
                historical_sales = historical_sales.sort_values('Date')

                # Calculate average growth rate
                historical_sales['Growth'] = historical_sales['TotalSales'].pct_change()
                avg_growth = historical_sales['Growth'].mean()

                # Get last date and sales value
                last_date = historical_sales['Date'].max()
                last_sales = historical_sales.loc[historical_sales['Date'] == last_date, 'TotalSales'].values[0]

                # Generate predictions
                predictions = []
                current_sales = last_sales

                for i in range(1, prediction_months + 1):
                    next_date = last_date + pd.DateOffset(months=i)
                    next_sales = current_sales * (1 + avg_growth)
                    predictions.append({
                        'Year': next_date.year,
                        'Month': next_date.month,
                        'PredictedSales': next_sales
                    })
                    current_sales = next_sales

                predictions_df = pd.DataFrame(predictions)
                return {"historical": historical_sales, "predictions": predictions_df}

            return "No historical sales data found for prediction."
        except Exception as e:
            return f"Error executing prediction query: {str(e)}"

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
            prediction_text += prediction_result["historical"].tail(5).to_string() + "\n\n"
            prediction_text += "Sales predictions for the next few months:\n"
            prediction_text += prediction_result["predictions"].to_string()

            # Generate explanation with Groq
            prompt = f"""You are an AI assistant that explains sales predictions.
Sales prediction data:
{prediction_text}
Product Category: {product_category if product_category else "All categories"}
Provide a brief explanation of these sales predictions. Label your response as an AI prediction.
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