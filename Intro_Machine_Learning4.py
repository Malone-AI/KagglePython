"""
    Random Forests
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)
mel_data = mel_data.dropna(axis = 0)
y = mel_data.Price
mel_features = ["Rooms","Bathroom","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude"]
X = mel_data[mel_features]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 0)
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X,train_y)
val_pre = forest_model.predict(val_X)
mae = mean_absolute_error(val_y,val_pre)
print(mae)