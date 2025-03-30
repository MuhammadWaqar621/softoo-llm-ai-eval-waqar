"""
Evaluation utilities for forecasting models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rich.console import Console
from rich.table import Table
import json
import os
from datetime import datetime

console = Console()


class ModelEvaluator:
    """Evaluates forecasting models."""

    def __init__(self, results_dir="results"):
        """
        Initialize model evaluator.

        Args:
            results_dir (str): Directory to store evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.metrics_history = {}

    def evaluate_prediction(self, y_true, y_pred):
        """
        Calculate evaluation metrics for predictions.

        Args:
            y_true (array): True values
            y_pred (array): Predicted values

        Returns:
            dict: Evaluation metrics
        """
        # Handle empty arrays or None values
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            return {
                "rmse": float('nan'),
                "mae": float('nan'),
                "r2": float('nan'),
                "mape": float('nan'),
                "smape": float('nan')
            }
            
        # Ensure arrays are the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage errors (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else float('nan')
        
        # Calculate symmetric MAPE (handles zero values better)
        denominator = np.abs(y_pred) + np.abs(y_true)
        mask = denominator != 0
        smape = 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) if np.any(mask) else float('nan')
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "smape": smape
        }
        
        return metrics

    def compare_models(self, actual, predictions_dict):
        """
        Compare multiple model predictions.

        Args:
            actual (array): Actual values
            predictions_dict (dict): Dictionary of model predictions

        Returns:
            DataFrame: Comparison metrics
        """
        comparison = []
        
        for model_name, predictions in predictions_dict.items():
            metrics = self.evaluate_prediction(actual, predictions)
            metrics["model"] = model_name
            comparison.append(metrics)
            
        return pd.DataFrame(comparison)

    def track_model_performance(self, category, model_type, metrics):
        """
        Track model performance over time.

        Args:
            category (str): Product category
            model_type (str): Model type
            metrics (dict): Performance metrics

        Returns:
            None
        """
        key = f"{category}_{model_type}"
        
        if key not in self.metrics_history:
            self.metrics_history[key] = []
            
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics_history[key].append(metrics)
        
        # Save history to file
        history_path = os.path.join(self.results_dir, f"{key}_history.json")
        
        try:
            # Clean metrics for JSON serialization
            serializable_metrics = []
            for m in self.metrics_history[key]:
                clean_metrics = {k: (float(v) if isinstance(v, np.float64) else v) for k, v in m.items()}
                serializable_metrics.append(clean_metrics)
                
            with open(history_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving metrics history: {e}[/red]")

    def get_model_history(self, category, model_type):
        """
        Get historical performance for a model.

        Args:
            category (str): Product category
            model_type (str): Model type

        Returns:
            DataFrame: Historical performance
        """
        key = f"{category}_{model_type}"
        history_path = os.path.join(self.results_dir, f"{key}_history.json")
        
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                    return pd.DataFrame(history)
            except Exception as e:
                console.print(f"[red]Error loading history: {e}[/red]")
                return pd.DataFrame()
        elif key in self.metrics_history:
            return pd.DataFrame(self.metrics_history[key])
            
        return pd.DataFrame()

    def calculate_prediction_intervals(self, model, X, confidence=0.95):
        """
        Calculate prediction intervals for forecasts.

        Args:
            model: Model object
            X (array): Input features
            confidence (float): Confidence level

        Returns:
            tuple: Lower and upper bounds
        """
        # This implementation is for random forest models
        if hasattr(model, "estimators_"):
            try:
                # Get predictions from all trees
                predictions = np.array([tree.predict(X) for tree in model.estimators_])
                
                # Calculate confidence intervals
                alpha = (1 - confidence) / 2
                lower_bound = np.percentile(predictions, 100 * alpha, axis=0)
                upper_bound = np.percentile(predictions, 100 * (1 - alpha), axis=0)
                
                return lower_bound, upper_bound
            except Exception as e:
                console.print(f"[yellow]Error calculating prediction intervals: {e}. Using simple approach.[/yellow]")
                
        # For other models, use a simpler approach
        try:
            y_pred = model.predict(X)
            # Estimate RMSE from model if available
            if hasattr(model, "mse_"):
                rmse = np.sqrt(model.mse_)
            elif hasattr(model, "_results") and hasattr(model._results, "mse"):
                rmse = np.sqrt(model._results.mse)
            else:
                # Use historical RMSE as a fallback
                rmse = 0.2 * np.mean(y_pred)  # Assume 20% error
            
            # Approximate intervals using RMSE
            z_score = 1.96  # 95% confidence
            lower_bound = y_pred - z_score * rmse
            upper_bound = y_pred + z_score * rmse
            
            # Ensure non-negative predictions for sales
            lower_bound = np.maximum(0, lower_bound)
            
            return lower_bound, upper_bound
        except Exception as e:
            console.print(f"[yellow]Error calculating simple prediction intervals: {e}[/yellow]")
            # Return dummy intervals if everything fails
            if isinstance(X, np.ndarray):
                y_pred = np.ones(X.shape[0])
            else:
                y_pred = np.array([1.0])
            return 0.8 * y_pred, 1.2 * y_pred

    def generate_forecast_report(self, category, model_type, historical_data, forecast_data, metrics):
        """
        Generate a comprehensive forecast report.

        Args:
            category (str): Product category
            model_type (str): Model type
            historical_data (DataFrame): Historical data
            forecast_data (DataFrame): Forecast data
            metrics (dict): Performance metrics

        Returns:
            str: Report content
        """
        # Create report content
        report = f"# Sales Forecast Report for {category}\n\n"
        report += f"## Model: {model_type}\n\n"
        
        # Add performance metrics
        report += "## Performance Metrics\n\n"
        report += f"- RMSE: {metrics['rmse']:.2f}\n"
        report += f"- MAE: {metrics['mae']:.2f}\n"
        report += f"- R²: {metrics['r2']:.4f}\n"
        if 'mape' in metrics and not np.isnan(metrics['mape']):
            report += f"- MAPE: {metrics['mape']:.2f}%\n"
        
        # Add forecast summary
        report += "\n## Forecast Summary\n\n"
        
        if not forecast_data.empty and 'Date' in forecast_data.columns and 'prediction' in forecast_data.columns:
            try:
                report += f"- Forecast Period: {forecast_data['Date'].min().strftime('%Y-%m-%d')} to {forecast_data['Date'].max().strftime('%Y-%m-%d')}\n"
                report += f"- Total Forecasted Sales: {forecast_data['prediction'].sum():.2f}\n"
                
                # Average monthly sales
                forecast_data['Year-Month'] = forecast_data['Date'].dt.strftime('%Y-%m')
                monthly_forecast = forecast_data.groupby('Year-Month').agg({'prediction': 'sum'})
                report += f"- Average Monthly Sales: {monthly_forecast['prediction'].mean():.2f}\n"
                
                # Compare with previous period if available
                if not historical_data.empty and 'Sales' in historical_data.columns:
                    last_period_sales = historical_data['Sales'].sum()
                    forecast_sales = forecast_data['prediction'].sum()
                    pct_change = (forecast_sales - last_period_sales) / last_period_sales * 100
                    report += f"- Expected Growth: {pct_change:.2f}%\n"
            except Exception as e:
                report += f"Error generating forecast metrics: {e}\n"
        else:
            report += "No forecast data available for summary.\n"
        
        # Save report to file
        report_path = os.path.join(self.results_dir, f"{category}_{model_type}_report.md")
        try:
            with open(report_path, "w") as f:
                f.write(report)
        except Exception as e:
            console.print(f"[yellow]Error saving report: {e}[/yellow]")
            
        return report

    def create_metrics_table(self, metrics_dict):
        """
        Create a rich table from metrics dictionary.

        Args:
            metrics_dict (dict): Dictionary of metrics

        Returns:
            Table: Rich table
        """
        table = Table(title="Model Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in metrics_dict.items():
            if metric in ["rmse", "mae"]:
                table.add_row(metric.upper(), f"{value:.2f}")
            elif metric == "r2":
                table.add_row("R²", f"{value:.4f}")
            elif metric in ["mape", "smape"]:
                table.add_row(metric.upper(), f"{value:.2f}%")
            elif not metric.startswith("_") and metric != "timestamp":  # Skip private attributes and timestamp
                table.add_row(metric, str(value))
                
        return table
        
    def evaluate_model_stability(self, category, model_type, time_periods=5):
        """
        Evaluate how stable a model's predictions are over time.
        
        Args:
            category (str): Product category
            model_type (str): Model type
            time_periods (int): Number of time periods to evaluate
            
        Returns:
            dict: Stability metrics
        """
        history = self.get_model_history(category, model_type)
        
        if history.empty or len(history) < 2:
            return {"stability": "Unknown", "metrics_variance": float('nan')}
            
        # Calculate variance of key metrics
        metric_variances = {}
        for metric in ['rmse', 'r2', 'mae']:
            if metric in history.columns:
                metric_variances[metric] = history[metric].var()
                
        # Calculate overall stability score (lower is better)
        if metric_variances:
            # Normalize variances by the mean of each metric
            normalized_variances = []
            for metric, variance in metric_variances.items():
                mean = history[metric].mean()
                if mean != 0:
                    normalized_variances.append(variance / abs(mean))
                    
            stability_score = sum(normalized_variances) / len(normalized_variances) if normalized_variances else float('nan')
            
            # Classify stability
            if stability_score < 0.05:
                stability = "Excellent"
            elif stability_score < 0.1:
                stability = "Good"
            elif stability_score < 0.2:
                stability = "Fair"
            else:
                stability = "Poor"
                
            return {
                "stability": stability,
                "stability_score": stability_score,
                "metric_variances": metric_variances
            }
        else:
            return {"stability": "Unknown", "metrics_variance": float('nan')}
            
    def evaluate_all_models(self):
        """
        Evaluate all models and find the best one for each category.
        
        Returns:
            DataFrame: Evaluation results for all models
        """
        results = []
        
        # Get all history files
        for filename in os.listdir(self.results_dir):
            if filename.endswith("_history.json"):
                try:
                    # Extract category and model type from filename
                    parts = filename.replace("_history.json", "").split("_")
                    if len(parts) >= 2:
                        model_type = parts[-1]
                        category = "_".join(parts[:-1])
                        
                        # Get latest metrics
                        history = self.get_model_history(category, model_type)
                        if not history.empty:
                            latest_metrics = history.iloc[-1].to_dict()
                            
                            # Add stability metrics
                            stability_metrics = self.evaluate_model_stability(category, model_type)
                            
                            # Combine all metrics
                            result = {
                                "category": category,
                                "model_type": model_type,
                                "rmse": latest_metrics.get("rmse", float('nan')),
                                "r2": latest_metrics.get("r2", float('nan')),
                                "stability": stability_metrics.get("stability", "Unknown"),
                                "timestamp": latest_metrics.get("timestamp", "Unknown")
                            }
                            results.append(result)
                except Exception as e:
                    console.print(f"[yellow]Error evaluating model {filename}: {e}[/yellow]")
                    
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=["category", "model_type", "rmse", "r2", "stability", "timestamp"])
            
    def find_best_models(self):
        """
        Find the best model for each category.
        
        Returns:
            dict: Best model for each category
        """
        all_models = self.evaluate_all_models()
        
        if all_models.empty:
            return {}
            
        best_models = {}
        
        # Group by category and find best model
        for category, group in all_models.groupby("category"):
            # Sort by R² (descending) and RMSE (ascending)
            sorted_models = group.sort_values(["r2", "rmse"], ascending=[False, True])
            
            if not sorted_models.empty:
                best_model = sorted_models.iloc[0]
                best_models[category] = {
                    "model_type": best_model["model_type"],
                    "rmse": best_model["rmse"],
                    "r2": best_model["r2"],
                    "stability": best_model["stability"]
                }
                
        return best_models