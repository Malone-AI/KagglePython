"""
    Model Validation

    The prediction error for each house is:
        error = actual - predicted
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)
# print(mel_data.describe())
# print(mel_data.head())
# print(mel_data.columns)

mel_data = mel_data.dropna(axis = 0)
# print(mel_data.describe())
# print(mel_data.head())
# print(mel_data.columns)

y = mel_data.Price
mel_features = ["Rooms","Bathroom","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude"]
X = mel_data[mel_features]

mel_model1 = DecisionTreeRegressor()
mel_model1.fit(X,y)
predicted_home_prices = mel_model1.predict(X)
# Calculate MAE
MAE = mean_absolute_error(y,predicted_home_prices)
print(MAE)

train_X,val_X,train_Y,val_Y = train_test_split(X,y,random_state = 0)
mel_model2 = DecisionTreeRegressor()
mel_model2.fit(train_X,train_Y)
val_predictions = mel_model2.predict(val_X)
MAE = mean_absolute_error(val_Y,val_predictions)
print(MAE)