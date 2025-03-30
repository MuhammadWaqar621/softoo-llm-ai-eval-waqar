"""
Visualization utilities for sales forecasting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import os


class ForecastVisualizer:
    """Visualization tools for forecasting data."""

    def __init__(self, output_dir="visualizations"):
        """
        Initialize forecast visualizer.

        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams.update({'figure.figsize': (12, 6)})

    def plot_forecast(self, historical_data, forecast_data, category=None, save_path=None):
        """
        Plot historical data and forecast.

        Args:
            historical_data (DataFrame): Historical sales data
            forecast_data (DataFrame): Forecast data
            category (str, optional): Product category
            save_path (str, optional): Path to save the plot

        Returns:
            str: Base64 encoded image or path to saved file
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to monthly data if not already
        if 'Date' in historical_data.columns and 'Date' in forecast_data.columns:
            # Group historical data by month
            historical_monthly = historical_data.copy()
            if 'Category' in historical_monthly.columns and category:
                historical_monthly = historical_monthly[historical_monthly['Category'] == category]
                
            # Convert dates to period for monthly grouping
            historical_monthly['YearMonth'] = historical_monthly['Date'].dt.to_period('M')
            historical_agg = historical_monthly.groupby('YearMonth').agg({'Sales': 'sum'}).reset_index()
            historical_agg['Date'] = historical_agg['YearMonth'].dt.to_timestamp()
            
            # Plot historical data
            ax.plot(historical_agg['Date'], historical_agg['Sales'], 
                   marker='o', linestyle='-', color='blue', label='Historical Sales')
            
            # Plot forecast data
            forecast_monthly = forecast_data.copy()
            forecast_monthly['YearMonth'] = forecast_monthly['Date'].dt.to_period('M')
            forecast_agg = forecast_monthly.groupby('YearMonth').agg({'prediction': 'sum'}).reset_index()
            forecast_agg['Date'] = forecast_agg['YearMonth'].dt.to_timestamp()
            
            ax.plot(forecast_agg['Date'], forecast_agg['prediction'], 
                   marker='o', linestyle='--', color='red', label='Forecast')
            
            # Add confidence intervals if available
            if 'lower_bound' in forecast_agg.columns and 'upper_bound' in forecast_agg.columns:
                ax.fill_between(forecast_agg['Date'], 
                                forecast_agg['lower_bound'], 
                                forecast_agg['upper_bound'],
                                color='red', alpha=0.2, label='95% Confidence Interval')
        else:
            # Fallback if date columns are not available
            ax.plot(range(len(historical_data)), historical_data['Sales'], 
                   marker='o', linestyle='-', color='blue', label='Historical Sales')
            ax.plot(range(len(historical_data), len(historical_data) + len(forecast_data)), 
                   forecast_data['prediction'], marker='o', linestyle='--', color='red', label='Forecast')
        
        # Add title and labels
        title = f"Sales Forecast for {category}" if category else "Sales Forecast"
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or return as base64
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Convert to base64 for display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{image_base64}"

    def plot_performance_metrics(self, metrics_history, category=None, model_type=None, save_path=None):
        """
        Plot model performance metrics over time.

        Args:
            metrics_history (DataFrame): History of model metrics
            category (str, optional): Product category
            model_type (str, optional): Model type
            save_path (str, optional): Path to save the plot

        Returns:
            str: Base64 encoded image or path to saved file
        """
        if not isinstance(metrics_history, pd.DataFrame) or metrics_history.empty:
            return None
            
        # Parse timestamp if available
        if 'timestamp' in metrics_history.columns:
            metrics_history['timestamp'] = pd.to_datetime(metrics_history['timestamp'])
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot RMSE
        if 'rmse' in metrics_history.columns:
            ax1.plot(metrics_history['timestamp'] if 'timestamp' in metrics_history.columns else range(len(metrics_history)), 
                    metrics_history['rmse'], marker='o', linestyle='-', color='red')
            ax1.set_title('RMSE Over Time')
            ax1.set_ylabel('RMSE')
            ax1.grid(True)
        
        # Plot R²
        if 'r2' in metrics_history.columns:
            ax2.plot(metrics_history['timestamp'] if 'timestamp' in metrics_history.columns else range(len(metrics_history)), 
                    metrics_history['r2'], marker='o', linestyle='-', color='blue')
            ax2.set_title('R² Over Time')
            ax2.set_ylabel('R²')
            ax2.grid(True)
            
        # Add overall title
        title = f"Model Performance Metrics for {category} ({model_type})" if category and model_type else "Model Performance Metrics"
        fig.suptitle(title, fontsize=16)
        
        # Save or return as base64
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Convert to base64 for display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{image_base64}"
            
    def plot_category_comparison(self, historical_data, forecast_data, categories=None, save_path=None):
        """
        Plot sales comparison across multiple categories.

        Args:
            historical_data (DataFrame): Historical sales data
            forecast_data (DataFrame): Forecast data
            categories (list, optional): List of categories to compare
            save_path (str, optional): Path to save the plot

        Returns:
            str: Base64 encoded image or path to saved file
        """
        if 'Category' not in historical_data.columns:
            return None
            
        if categories is None:
            categories = historical_data['Category'].unique()
            
        # Limit to max 5 categories for readability
        if len(categories) > 5:
            categories = categories[:5]
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert to monthly data 
        historical_monthly = historical_data.copy()
        historical_monthly['YearMonth'] = pd.to_datetime(historical_monthly['Date']).dt.to_period('M')
        
        # Plot each category
        for category in categories:
            category_data = historical_monthly[historical_monthly['Category'] == category]
            category_agg = category_data.groupby('YearMonth').agg({'Sales': 'sum'}).reset_index()
            category_agg['Date'] = category_agg['YearMonth'].dt.to_timestamp()
            
            ax.plot(category_agg['Date'], category_agg['Sales'], 
                   marker='o', linestyle='-', label=f'{category} (Historical)')
            
            # Add forecast if available
            if 'Category' in forecast_data.columns:
                category_forecast = forecast_data[forecast_data['Category'] == category]
                if not category_forecast.empty:
                    forecast_monthly = category_forecast.copy()
                    forecast_monthly['YearMonth'] = pd.to_datetime(forecast_monthly['Date']).dt.to_period('M')
                    forecast_agg = forecast_monthly.groupby('YearMonth').agg({'prediction': 'sum'}).reset_index()
                    forecast_agg['Date'] = forecast_agg['YearMonth'].dt.to_timestamp()
                    
                    ax.plot(forecast_agg['Date'], forecast_agg['prediction'], 
                           marker='o', linestyle='--', label=f'{category} (Forecast)')
        
        # Add title and labels
        ax.set_title("Category Sales Comparison", fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or return as base64
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Convert to base64 for display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{image_base64}"

    def plot_seasonal_patterns(self, historical_data, category=None, save_path=None):
        """
        Plot seasonal patterns in sales data.

        Args:
            historical_data (DataFrame): Historical sales data
            category (str, optional): Product category
            save_path (str, optional): Path to save the plot

        Returns:
            str: Base64 encoded image or path to saved file
        """
        # Filter by category if specified
        data = historical_data.copy()
        if category and 'Category' in data.columns:
            data = data[data['Category'] == category]
            
        if data.empty:
            return None
            
        # Ensure date column is datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract month and day of week
        data['Month'] = data['Date'].dt.month
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot monthly pattern
        monthly_data = data.groupby('Month')['Sales'].mean().reset_index()
        sns.barplot(x='Month', y='Sales', data=monthly_data, ax=ax1)
        ax1.set_title('Monthly Sales Pattern')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Sales')
        ax1.set_xticks(range(12))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Plot day of week pattern
        daily_data = data.groupby('DayOfWeek')['Sales'].mean().reset_index()
        sns.barplot(x='DayOfWeek', y='Sales', data=daily_data, ax=ax2)
        ax2.set_title('Day of Week Sales Pattern')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Sales')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add overall title
        title = f"Seasonal Sales Patterns for {category}" if category else "Seasonal Sales Patterns"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the title
        
        # Save or return as base64
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Convert to base64 for display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{image_base64}"