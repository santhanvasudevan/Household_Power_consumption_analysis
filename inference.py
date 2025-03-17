import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model
def load_model(model_type):
    model_filename = f"trained_model_{model_type}.pkl"
    with open(model_filename, "rb") as file:
        model = pickle.load(file)
    return model

# Load and preprocess new data for inference
def preprocess_new_data(filepath, scaler_path):
    df = pd.read_csv(filepath, sep=';', parse_dates=[['Date', 'Time']], na_values=['?'])
    df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
    df.set_index('Datetime', inplace=True)
    
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek + 1
    df['time_float'] = df.index.hour + df.index.minute / 60.0

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)

    # Drop target column if it exists
    if 'Global_active_power' in df.columns:
        df.drop(columns=['Global_active_power'], inplace=True)

    # Load the scaler
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Apply scaling
    X_scaled = scaler.transform(df)
    
    return X_scaled, df.index  # Return index to align predictions

# Make predictions
def make_predictions(model, X, index):
    predictions = model.predict(X)
    return pd.DataFrame({'Datetime': index, 'Predicted_Global_active_power': predictions})

if __name__ == "__main__":
    model_types = ["random_forest", "gradient_boosting", "linear_regression"]
    X_new, index = preprocess_new_data("new_data.csv", "scaler.pkl")

    # Predict using each model and save results
    for model_type in model_types:
        print(f"Making predictions with {model_type} model...")
        model = load_model(model_type)
        predictions = make_predictions(model, X_new, index)
        
        # Save predictions separately for each model
        output_filename = f"predictions_{model_type}.csv"
        predictions.to_csv(output_filename, index=False)
        
        print(f"Saved predictions to {output_filename}")
    
    print("Inference completed for all models.")
