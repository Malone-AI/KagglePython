"""
    Your First Machine Learning Model
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)
print(mel_data.columns)
print(mel_data.describe())
print(mel_data.head())

# Drop nan
mel_data = mel_data.dropna(axis = 0)

# Select The Prediction Target
y = mel_data.Price
print(y)

# Choosing The Features
mel_features = ["Rooms","Bathroom", "Landsize","Lattitude","Longtitude"]

X = mel_data[mel_features]
print(X.describe())
print(X.head())

# Building My Model
mel_model = DecisionTreeRegressor(random_state = 1)
mel_model.fit(X,y)

print("Predictions:")
print(mel_model.predict(X.head()))
print("Real Values:")
print(mel_data.head()["Price"])