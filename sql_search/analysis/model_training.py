"""
Model training pipeline for AdventureWorksLT database.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rich.console import Console
from rich.table import Table
from datetime import datetime, timedelta
from sql_search.analysis.ml_models import ModelManager, TimeSeriesModel

console = Console()


class ModelTrainer:
    """Trains and evaluates forecasting models using AdventureWorksLT sales data."""

    def __init__(self, db_connection, model_manager=None):
        """
        Initialize model trainer.

        Args:
            db_connection: DatabaseConnection instance
            model_manager (ModelManager, optional): Model manager
        """
        self.db_connection = db_connection
        self.model_manager = model_manager or ModelManager()
        self.training_metrics = {}
        self.testing_metrics = {}
        self.has_trained_models = False

    def get_training_data(self):
        """
        Get training data from the AdventureWorksLT database.

        Returns:
            DataFrame: Training data
        """
        if not self.db_connection.is_connected():
            return None

        try:
            # Use actual sales data from AdventureWorksLT
            query = """
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

            # Execute the query - now returns a DataFrame directly
            result = self.db_connection.execute_query(query)
            
            # Check if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    # Ensure date column is datetime
                    if 'SalesDate' in result.columns:
                        result["SalesDate"] = pd.to_datetime(result["SalesDate"])
                        # Rename columns to match model expectations
                        result = result.rename(columns={"SalesDate": "Date"})
                    return result
                else:
                    console.print("[yellow]No data found in database. Generating demo data for training.[/yellow]")
                    return self._generate_demo_data()
            elif isinstance(result, str):
                # Handle error message
                console.print(f"[red]Error getting training data: {result}[/red]")
                console.print("[yellow]Falling back to demo data.[/yellow]")
                return self._generate_demo_data()
            else:
                console.print("[yellow]Unexpected result type. Falling back to demo data.[/yellow]")
                return self._generate_demo_data()
                
        except Exception as e:
            console.print(f"[red]Error getting training data: {e}[/red]")
            console.print("[yellow]Generating demo data for training.[/yellow]")
            return self._generate_demo_data()
    
    def _generate_demo_data(self):
        """
        Generate demo data for training if no data exists in database.
        
        Returns:
            DataFrame: Generated demo data
        """
        console.print("[yellow]Generating demo data for model training...[/yellow]")
        
        # Starting date (one year ago)
        start_date = datetime.now() - timedelta(days=365)
        
        # Generate date range
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        # Default categories
        categories = ["Mountain Bikes", "Road Bikes", "Clothing", "Accessories"]
        
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
        
        console.print(f"[green]Generated demo data with {len(data)} records for {len(categories)} categories[/green]")
        return pd.DataFrame(data)

    def train_and_evaluate(self, df=None, categories=None, model_types=None, test_size=0.2):
        """
        Train and evaluate models on AdventureWorksLT sales data.

        Args:
            df (DataFrame, optional): Training data
            categories (list, optional): Product categories
            model_types (list, optional): Types of models to train
            test_size (float): Proportion of data to use for testing

        Returns:
            tuple: Training metrics, testing metrics
        """
        if df is None:
            df = self.get_training_data()
            
        if df is None or df.empty:
            console.print("[red]No training data available from AdventureWorksLT database.[/red]")
            return None, None
            
        if categories is None:
            categories = df["Category"].unique()
            
        if model_types is None:
            model_types = ["random_forest", "arima"]

        # Initialize metrics dictionaries
        self.training_metrics = {category: {model_type: {} for model_type in model_types} for category in categories}
        self.testing_metrics = {category: {model_type: {} for model_type in model_types} for category in categories}
        
        # Flag to track if any models were successfully trained
        any_models_trained = False

        for category in categories:
            console.print(f"[green]Training models for category: {category}[/green]")
            category_data = df[df["Category"] == category].sort_values("Date")
            
            # Skip if not enough data
            if len(category_data) < 30:
                console.print(f"[yellow]Not enough data for category {category}. Skipping.[/yellow]")
                continue
            
            # Split data into training and testing
            split_idx = int(len(category_data) * (1 - test_size))
            train_data = category_data.iloc[:split_idx]
            test_data = category_data.iloc[split_idx:]
            
            for model_type in model_types:
                try:
                    console.print(f"  Training {model_type} model...")
                    
                    # Train model
                    model = TimeSeriesModel(model_type=model_type)
                    model.train(train_data)
                    
                    # Evaluate on training data
                    train_metrics = self._evaluate_model(model, train_data)
                    self.training_metrics[category][model_type] = train_metrics
                    
                    # Evaluate on testing data
                    test_metrics = self._evaluate_model(model, test_data)
                    self.testing_metrics[category][model_type] = test_metrics
                    
                    # Save model
                    model_path = self.model_manager.get_model_path(category, model_type)
                    model.save(model_path)
                    self.model_manager.models[(category, model_type)] = model
                    
                    # Print metrics
                    console.print(f"    Training RMSE: {train_metrics['rmse']:.2f}")
                    console.print(f"    Testing RMSE: {test_metrics['rmse']:.2f}")
                    console.print(f"    Testing R²: {test_metrics['r2']:.4f}")
                    
                    # Mark that we successfully trained at least one model
                    any_models_trained = True
                    self.has_trained_models = True
                    
                except Exception as e:
                    console.print(f"[red]Error training {model_type} model for {category}: {e}[/red]")
                
        if not any_models_trained:
            console.print("[yellow]No models were trained due to insufficient data. Using demo data instead.[/yellow]")
            # If no actual categories had enough data, use our demo data directly
            return self.train_on_demo_data(model_types)
                
        return self.training_metrics, self.testing_metrics
    
    def train_on_demo_data(self, model_types=None):
        """
        Train models on generated demo data.
        
        Args:
            model_types (list, optional): Types of models to train
            
        Returns:
            tuple: Training metrics, testing metrics
        """
        console.print("[yellow]Training on generated demo data...[/yellow]")
        
        # Generate demo data with predefined categories
        demo_data = self._generate_demo_data()
        
        # Use default model types if not specified
        if model_types is None:
            model_types = ["random_forest", "arima"]
            
        # Get categories from demo data
        categories = demo_data["Category"].unique()
        
        # Reset metrics
        self.training_metrics = {category: {model_type: {} for model_type in model_types} for category in categories}
        self.testing_metrics = {category: {model_type: {} for model_type in model_types} for category in categories}
        
        # Train and evaluate for each category
        test_size = 0.2
        any_models_trained = False
        
        for category in categories:
            console.print(f"[green]Training models for demo category: {category}[/green]")
            category_data = demo_data[demo_data["Category"] == category].sort_values("Date")
            
            # Split data into training and testing
            split_idx = int(len(category_data) * (1 - test_size))
            train_data = category_data.iloc[:split_idx]
            test_data = category_data.iloc[split_idx:]
            
            for model_type in model_types:
                try:
                    console.print(f"  Training {model_type} model...")
                    
                    # Train model
                    model = TimeSeriesModel(model_type=model_type)
                    model.train(train_data)
                    
                    # Evaluate on training data
                    train_metrics = self._evaluate_model(model, train_data)
                    self.training_metrics[category][model_type] = train_metrics
                    
                    # Evaluate on testing data
                    test_metrics = self._evaluate_model(model, test_data)
                    self.testing_metrics[category][model_type] = test_metrics
                    
                    # Save model
                    model_path = self.model_manager.get_model_path(category, model_type)
                    model.save(model_path)
                    self.model_manager.models[(category, model_type)] = model
                    
                    # Print metrics
                    console.print(f"    Training RMSE: {train_metrics['rmse']:.2f}")
                    console.print(f"    Testing RMSE: {test_metrics['rmse']:.2f}")
                    console.print(f"    Testing R²: {test_metrics['r2']:.4f}")
                    
                    # Mark that we successfully trained at least one model
                    any_models_trained = True
                    self.has_trained_models = True
                    
                except Exception as e:
                    console.print(f"[red]Error training {model_type} model for {category}: {e}[/red]")
        
        if not any_models_trained:
            console.print("[red]Failed to train any models, even on demo data.[/red]")
            
        return self.training_metrics, self.testing_metrics

    def _evaluate_model(self, model, data):
        """
        Evaluate a model on the given data.

        Args:
            model (TimeSeriesModel): Model to evaluate
            data (DataFrame): Data to evaluate on

        Returns:
            dict: Evaluation metrics
        """
        # Get feature columns and target
        X, y_true = model.prepare_features(data)
        
        if model.model_type in ["random_forest", "linear_regression"]:
            # Use model to predict
            y_pred = model.model.predict(X)
        elif model.model_type in ["arima", "exp_smoothing"]:
            # For time series models, use the in-sample predictions
            y_pred = model.model.fittedvalues.values
            
            # Handle length mismatch if any
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def cross_validate(self, df=None, categories=None, model_types=None, n_splits=5):
        """
        Perform time series cross-validation.

        Args:
            df (DataFrame, optional): Training data
            categories (list, optional): Product categories
            model_types (list, optional): Types of models to train
            n_splits (int): Number of splits for cross-validation

        Returns:
            dict: Cross-validation metrics
        """
        if df is None:
            df = self.get_training_data()
            
        if df is None or df.empty:
            console.print("[red]No training data available from AdventureWorksLT database.[/red]")
            return None
            
        if categories is None:
            categories = df["Category"].unique()
            
        if model_types is None:
            model_types = ["random_forest", "arima"]

        # Initialize cross-validation metrics
        cv_metrics = {category: {model_type: [] for model_type in model_types} for category in categories}

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        any_models_evaluated = False

        for category in categories:
            console.print(f"[green]Cross-validating models for category: {category}[/green]")
            category_data = df[df["Category"] == category].sort_values("Date")
            
            # Skip if not enough data
            if len(category_data) < 30:
                console.print(f"[yellow]Not enough data for category {category}. Skipping.[/yellow]")
                continue
                
            for model_type in model_types:
                console.print(f"  Cross-validating {model_type} model...")
                
                fold_metrics = []
                for train_idx, test_idx in tscv.split(category_data):
                    train_data = category_data.iloc[train_idx]
                    test_data = category_data.iloc[test_idx]
                    
                    # Skip if too little data
                    if len(train_data) < 30 or len(test_data) < 10:
                        continue
                    
                    # Train model
                    model = TimeSeriesModel(model_type=model_type)
                    model.train(train_data)
                    
                    # Evaluate on testing data
                    test_metrics = self._evaluate_model(model, test_data)
                    fold_metrics.append(test_metrics)
                
                # Calculate average metrics
                if fold_metrics:
                    avg_rmse = sum(m["rmse"] for m in fold_metrics) / len(fold_metrics)
                    avg_mae = sum(m["mae"] for m in fold_metrics) / len(fold_metrics)
                    avg_r2 = sum(m["r2"] for m in fold_metrics) / len(fold_metrics)
                    
                    cv_metrics[category][model_type] = {
                        "avg_rmse": avg_rmse,
                        "avg_mae": avg_mae,
                        "avg_r2": avg_r2,
                        "fold_metrics": fold_metrics
                    }
                    
                    console.print(f"    Average CV RMSE: {avg_rmse:.2f}")
                    console.print(f"    Average CV R²: {avg_r2:.4f}")
                    
                    any_models_evaluated = True
        
        if not any_models_evaluated:
            console.print("[yellow]No models could be cross-validated due to insufficient data. Consider using demo data.[/yellow]")
            # Return empty but valid metrics
            return cv_metrics
                
        return cv_metrics

    def display_metrics(self):
        """
        Display training and testing metrics.

        Returns:
            Table: Rich table with metrics
        """
        table = Table(title="Model Performance Metrics")
        table.add_column("Category", style="cyan")
        table.add_column("Model Type", style="magenta")
        table.add_column("Training RMSE", style="green")
        table.add_column("Testing RMSE", style="red")
        table.add_column("R²", style="blue")
        
        # Check if any models were actually trained
        metrics_available = False
        
        for category in self.training_metrics:
            for model_type in self.training_metrics[category]:
                # Only add rows for categories/models that have metrics
                if self.training_metrics[category][model_type] and 'rmse' in self.training_metrics[category][model_type]:
                    train_metrics = self.training_metrics[category][model_type]
                    
                    # Check if we have test metrics for this model too
                    if self.testing_metrics.get(category, {}).get(model_type, {}).get('rmse') is not None:
                        test_metrics = self.testing_metrics[category][model_type]
                        
                        table.add_row(
                            category,
                            model_type,
                            f"{train_metrics['rmse']:.2f}",
                            f"{test_metrics['rmse']:.2f}",
                            f"{test_metrics['r2']:.4f}"
                        )
                        metrics_available = True
        
        if not metrics_available:
            console.print("[yellow]No metrics available. No models were successfully trained.[/yellow]")
            # Create a simple table with a message
            empty_table = Table(title="Model Training Status")
            empty_table.add_column("Status", style="yellow")
            empty_table.add_row("No models were trained due to insufficient data.")
            return empty_table
            
        return table

    def find_best_model(self, category):
        """
        Find the best model for a specific category.

        Args:
            category (str): Product category

        Returns:
            tuple: Best model type, metrics
        """
        if not self.testing_metrics or category not in self.testing_metrics:
            console.print(f"[yellow]No metrics available for category: {category}[/yellow]")
            return None, None
            
        best_model = None
        best_metrics = None
        best_r2 = -float('inf')
        
        for model_type, metrics in self.testing_metrics[category].items():
            # Check if metrics contain r2 score
            if 'r2' in metrics and metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = model_type
                best_metrics = metrics
                
        return best_model, best_metrics