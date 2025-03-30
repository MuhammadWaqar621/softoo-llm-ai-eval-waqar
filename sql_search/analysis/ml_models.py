"""
Machine learning models for forecasting.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import os
from datetime import datetime, timedelta


class TimeSeriesModel:
    """Base class for time series forecasting models."""

    def __init__(self, model_type="random_forest"):
        """
        Initialize time series model.

        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None

    def prepare_features(self, df, target_column="Sales", date_column="Date"):
        """
        Prepare features for time series forecasting.

        Args:
            df (DataFrame): Input data
            target_column (str): Target column name
            date_column (str): Date column name

        Returns:
            tuple: X features, y target
        """
        # Create a copy of the dataframe
        data = df.copy()

        # Ensure date column is datetime
        if date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column])

        # Extract date features
        if date_column in data.columns:
            data["year"] = data[date_column].dt.year
            data["month"] = data[date_column].dt.month
            data["day"] = data[date_column].dt.day
            data["dayofweek"] = data[date_column].dt.dayofweek
            data["quarter"] = data[date_column].dt.quarter
            data["dayofyear"] = data[date_column].dt.dayofyear

        # Create lag features (previous values)
        for i in range(1, 4):  # Create 3 lag features
            data[f"{target_column}_lag_{i}"] = data[target_column].shift(i)

        # Create rolling window features
        data[f"{target_column}_rolling_mean_7"] = data[target_column].rolling(window=7).mean()
        data[f"{target_column}_rolling_mean_30"] = data[target_column].rolling(window=30).mean()

        # Handle missing values
        data = data.dropna()

        # Exclude date column and target from features
        feature_cols = [col for col in data.columns if col != target_column and col != date_column]
        self.feature_columns = feature_cols
        self.target_column = target_column

        # Scale features
        X = self.scaler.fit_transform(data[feature_cols])
        y = data[target_column].values

        return X, y

    def train(self, df, target_column="Sales", date_column="Date"):
        """
        Train the forecasting model.

        Args:
            df (DataFrame): Training data
            target_column (str): Target column name
            date_column (str): Date column name

        Returns:
            self: Trained model
        """
        X, y = self.prepare_features(df, target_column, date_column)

        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        elif self.model_type == "linear_regression":
            self.model = LinearRegression()
            self.model.fit(X, y)
        elif self.model_type == "arima":
            # Group by date and calculate mean for ARIMA
            ts_data = df.groupby(date_column)[target_column].mean()
            # Use auto_arima to find optimal parameters or set manually
            try:
                self.model = ARIMA(ts_data, order=(5, 1, 1))
                self.model = self.model.fit()
            except Exception as e:
                print(f"Error fitting ARIMA model: {e}. Falling back to RandomForest.")
                self.model_type = "random_forest"
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X, y)
        elif self.model_type == "exp_smoothing":
            # Group by date and calculate mean for Exponential Smoothing
            ts_data = df.groupby(date_column)[target_column].mean()
            try:
                self.model = ExponentialSmoothing(
                    ts_data, seasonal_periods=12, trend="add", seasonal="add"
                )
                self.model = self.model.fit()
            except Exception as e:
                print(f"Error fitting Exponential Smoothing model: {e}. Falling back to RandomForest.")
                self.model_type = "random_forest"
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X, y)

        return self

    def predict(self, future_dates, last_data):
        """
        Generate predictions for future dates.

        Args:
            future_dates (list): List of future dates to predict
            last_data (DataFrame): Recent data to use for prediction

        Returns:
            DataFrame: Predictions
        """
        predictions = []

        if self.model_type in ["random_forest", "linear_regression"]:
            # Create a dataframe for future dates
            future_df = pd.DataFrame({"Date": future_dates})
            future_df["year"] = future_df["Date"].dt.year
            future_df["month"] = future_df["Date"].dt.month
            future_df["day"] = future_df["Date"].dt.day
            future_df["dayofweek"] = future_df["Date"].dt.dayofweek
            future_df["quarter"] = future_df["Date"].dt.quarter
            future_df["dayofyear"] = future_df["Date"].dt.dayofyear

            # Get the most recent values for lag features
            last_values = last_data[self.target_column].iloc[-3:].values if len(last_data) >= 3 else np.array([0, 0, 0])
            
            for i, date in enumerate(future_dates):
                current_df = future_df[future_df["Date"] == date].copy()
                
                # Add lag features
                for j in range(1, 4):
                    idx = -j
                    if i >= j:
                        # Use previously predicted values
                        current_df[f"{self.target_column}_lag_{j}"] = predictions[i-j]["prediction"]
                    else:
                        # Use actual past values
                        if abs(idx) <= len(last_values):
                            current_df[f"{self.target_column}_lag_{j}"] = last_values[idx]
                        else:
                            current_df[f"{self.target_column}_lag_{j}"] = 0
                
                # Set rolling means (simplified)
                if len(last_data) >= 7:
                    current_df[f"{self.target_column}_rolling_mean_7"] = last_data[self.target_column].iloc[-7:].mean()
                else:
                    current_df[f"{self.target_column}_rolling_mean_7"] = last_data[self.target_column].mean() if not last_data.empty else 0
                    
                if len(last_data) >= 30:
                    current_df[f"{self.target_column}_rolling_mean_30"] = last_data[self.target_column].iloc[-30:].mean()
                else:
                    current_df[f"{self.target_column}_rolling_mean_30"] = last_data[self.target_column].mean() if not last_data.empty else 0
                
                # Ensure all feature columns exist
                for col in self.feature_columns:
                    if col not in current_df.columns:
                        current_df[col] = 0
                
                # Scale features
                try:
                    X_future = self.scaler.transform(current_df[self.feature_columns])
                    
                    # Predict
                    prediction = float(self.model.predict(X_future)[0])
                    
                    # Ensure prediction is not negative
                    prediction = max(0, prediction)
                except Exception as e:
                    print(f"Error during prediction: {e}. Using simple trend prediction.")
                    # Fallback to simple trend prediction
                    if i > 0:
                        prediction = predictions[i-1]["prediction"] * 1.01  # Assume 1% growth
                    else:
                        prediction = last_data[self.target_column].mean() if not last_data.empty else 1000
                
                predictions.append({
                    "Date": date,
                    "prediction": prediction
                })
                
        elif self.model_type == "arima":
            try:
                # Use ARIMA's built-in forecast function
                forecast = self.model.forecast(steps=len(future_dates))
                for i, date in enumerate(future_dates):
                    predictions.append({
                        "Date": date,
                        "prediction": max(0, forecast[i])  # Ensure non-negative
                    })
            except Exception as e:
                print(f"Error in ARIMA prediction: {e}. Using simple trend prediction.")
                # Fallback to simple trend
                base_value = last_data[self.target_column].mean() if not last_data.empty else 1000
                for i, date in enumerate(future_dates):
                    prediction = base_value * (1 + 0.01 * i)  # 1% growth per step
                    predictions.append({
                        "Date": date,
                        "prediction": prediction
                    })
                
        elif self.model_type == "exp_smoothing":
            try:
                # Use Exponential Smoothing's forecast
                forecast = self.model.forecast(steps=len(future_dates))
                for i, date in enumerate(future_dates):
                    predictions.append({
                        "Date": date,
                        "prediction": max(0, forecast[i])  # Ensure non-negative
                    })
            except Exception as e:
                print(f"Error in Exponential Smoothing prediction: {e}. Using simple trend prediction.")
                # Fallback to simple trend
                base_value = last_data[self.target_column].mean() if not last_data.empty else 1000
                for i, date in enumerate(future_dates):
                    prediction = base_value * (1 + 0.01 * i)  # 1% growth per step
                    predictions.append({
                        "Date": date,
                        "prediction": prediction
                    })

        return pd.DataFrame(predictions)

    def save(self, path):
        """
        Save the model to disk.

        Args:
            path (str): Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                "model_type": self.model_type,
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column
            }
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, path):
        """
        Load a model from disk.

        Args:
            path (str): Path to load the model from

        Returns:
            TimeSeriesModel: Loaded model
        """
        try:
            model_data = joblib.load(path)
            instance = cls(model_type=model_data["model_type"])
            instance.model = model_data["model"]
            instance.scaler = model_data["scaler"]
            instance.feature_columns = model_data["feature_columns"]
            instance.target_column = model_data["target_column"]
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


class ModelManager:
    """Manages multiple forecasting models."""

    def __init__(self, models_dir="models"):
        """
        Initialize model manager.

        Args:
            models_dir (str): Directory to store models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}

    def get_model_path(self, category, model_type):
        """
        Get path for a model.

        Args:
            category (str): Product category
            model_type (str): Type of model

        Returns:
            str: Model path
        """
        # Clean category name for filename
        clean_category = category.replace(" ", "_").replace("/", "_").lower()
        return os.path.join(self.models_dir, f"{clean_category}_{model_type}.joblib")

    def train_models(self, df, categories=None, model_types=None):
        """
        Train models for different categories.

        Args:
            df (DataFrame): Training data
            categories (list, optional): Product categories
            model_types (list, optional): Types of models to train

        Returns:
            dict: Dictionary of trained models
        """
        if categories is None:
            categories = df["Category"].unique()
            
        if model_types is None:
            model_types = ["random_forest", "arima"]

        for category in categories:
            category_data = df[df["Category"] == category]
            
            # Skip if not enough data
            if len(category_data) < 30:
                print(f"Not enough data for category {category}. Skipping.")
                continue
                
            for model_type in model_types:
                print(f"Training {model_type} model for {category}...")
                
                model = TimeSeriesModel(model_type=model_type)
                model.train(category_data)
                
                model_path = self.get_model_path(category, model_type)
                model.save(model_path)
                
                self.models[(category, model_type)] = model
                
        return self.models

    def load_model(self, category, model_type):
        """
        Load a model for a specific category and type.

        Args:
            category (str): Product category
            model_type (str): Type of model

        Returns:
            TimeSeriesModel: Loaded model
        """
        key = (category, model_type)
        
        if key in self.models:
            return self.models[key]
            
        model_path = self.get_model_path(category, model_type)
        
        if os.path.exists(model_path):
            model = TimeSeriesModel.load(model_path)
            self.models[key] = model
            return model
            
        return None

    def predict(self, category, model_type, future_dates, last_data):
        """
        Generate predictions using a specific model.

        Args:
            category (str): Product category
            model_type (str): Type of model
            future_dates (list): List of future dates to predict
            last_data (DataFrame): Recent data to use for prediction

        Returns:
            DataFrame: Predictions
        """
        model = self.load_model(category, model_type)
        
        if model is None:
            return None
            
        return model.predict(future_dates, last_data)