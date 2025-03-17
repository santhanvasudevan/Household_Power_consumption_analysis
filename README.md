# Household Power Consumption Prediction

This repository contains code for training and evaluating multiple machine learning models (Random Forest, Gradient Boosting, and Linear Regression) to predict household power consumption.

## 📌 Project Overview
The goal of this project is to analyze and predict `Global_active_power` using historical household power consumption data. The dataset is preprocessed, engineered with additional features, and used to train multiple regression models.

## 📂 Folder Structure
```
📦 project-folder
├── 📂 data                  # Raw and processed data
├── 📂 models                # Saved trained models
├── 📂 notebooks             # Jupyter notebooks for analysis
├── 📜 train.py              # Main script to load data, preprocess & train models
├── 📜 inference.py          # Script for making predictions on new data
├── 📜 requirements.txt      # List of dependencies
├── 📜 README.md             # Project documentation
```

## 📊 Dataset
The dataset used for this project is `household_power_consumption.txt`, which contains:
- `Global_active_power`
- `Voltage`
- `Sub_metering_1, Sub_metering_2, Sub_metering_3`
- Timestamps (`Date`, `Time`)

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Models
Run the following script to load, preprocess & train the models:
```bash
python train.py
```

### 4️⃣ Perform Inference on New Data
Once trained, use `inference.py` to make predictions(new data should be in this file: new_data.csv):
```bash
python inference.py
```

## 🔍 Features & Preprocessing
The preprocessing steps include:
- Handling missing values
- Checking correlation matrix
- Extracting time-based features (hour, minutes, day of the week)
- Creating new features:
  - `time_float` (combining hour and minutes)
  - `is_peak_hour`
  - `Global_active_power_daily_avg`
  - `Global_active_power_rolling` (rolling averages)
- Checking for outliers

## 🚀 Models Used
The following regression models are trained and compared:
- ✅ **Linear Regression**
- ✅ **Gradient Boosting Regressor**
- ✅ **Random Forest Regressor**

## 📈 Evaluation Metrics
The models are evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

## 🛠 Future Improvements
- Fine-tuning hyperparameters
- Feature selection and dimensionality reduction
- Deploying the model as an API for real-time predictions

## 📜 License
MIT License. See `LICENSE` file for details.


