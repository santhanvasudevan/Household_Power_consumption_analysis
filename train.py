import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
def load_data(filepath):
    """
    Loads and preprocesses the dataset.
    - Combines 'Date' and 'Time' columns into a datetime index.
    - Extracts additional time-based features.
    - Handles missing values and converts numeric columns.
    """
    df = pd.read_csv(filepath, sep=';', na_values=['?'], parse_dates=[['Date', 'Time']])
    df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
    df.set_index('Datetime', inplace=True)

    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek + 1
    df['time_float'] = df.index.hour + df.index.minute / 60.0

    df.dropna(subset=['Global_active_power'], inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# Exploratory Data Analysis
def explore_data(df):
    """Displays basic dataset insights and visualizations."""
    print("Data Head:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

# Feature Engineering
def preprocess_data(df):
    """
    Prepares data for training.
    - Selects target variable.
    - Splits into training & testing sets.
    - Applies standard scaling.
    """
    y = df['Global_active_power']
    X = df.drop('Global_active_power', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, scaler

# Model Training
def train_model(X_train, y_train, model_type="random_forest"):
    """
    Trains a machine learning model based on the specified type.
    Supports:
    - Random Forest
    - Linear Regression
    - Gradient Boosting
    """
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose from 'random_forest', 'linear_regression', 'gradient_boosting'.")

    model.fit(X_train, y_train)
    return model

# Feature Importance for Tree-Based Models
def feature_importance(model, feature_names):
    """Displays feature importance for Random Forest or Gradient Boosting models."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        print("\nFeature Importance Ranking:")
        print(feature_importance_df)

        plt.figure(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title("Feature Importance")
        plt.show()

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """Evaluates the model using MAE, RMSE, and R² metrics."""
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.legend()
    plt.title("Actual vs Predicted Energy Consumption")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load and explore dataset
    df = load_data(r"data\household_power_consumption.txt")
    explore_data(df)

    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)

    # Save the scaler
    with open("models\\scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Train and evaluate models
    models = ["linear_regression", "gradient_boosting", "random_forest"]
    # Dictionary to store trained models
    trained_models = {}
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        model = train_model(X_train, y_train, model_type)
        trained_models[model_type] = model
        evaluate_model(model, X_test, y_test)
    
        # Save each model with a unique filename
        model_filename = f"models\\trained_model_{model_type}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
    
        print(f"Saved {model_type} model as {model_filename}")


    # Feature importance (only for tree-based models)
    feature_importance(trained_models["random_forest"], feature_names)

    
